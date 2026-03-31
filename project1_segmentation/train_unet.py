import os
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, NormalizeIntensity, EnsureChannelFirst
import matplotlib.pyplot as plt

# ── Dataset ──────────────────────────────────────────────────────────────────
class HeartDataset(Dataset):
    def __init__(self, patient_ids, nifti_dir, ts_dir, augment=False):
        self.samples = []
        self.augment = augment
        
        for pid in patient_ids:
            ct_path = os.path.join(nifti_dir, f"patient_{pid}.nii.gz")
            mask_path = os.path.join(ts_dir, f"patient_{pid}", "heart.nii.gz")
            
            if not os.path.exists(ct_path) or not os.path.exists(mask_path):
                continue
                
            ct_arr = sitk.GetArrayFromImage(sitk.ReadImage(ct_path)).astype(np.float32)
            mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.uint8)
            
            # Cardiac window
            ct_arr = np.clip(ct_arr, -400, 800)
            ct_arr = (ct_arr + 400) / 1200.0
            
            # Use ALL slices, not just heart slices
            for s in range(ct_arr.shape[0]):
                self.samples.append((
                    ct_arr[s][np.newaxis].astype(np.float32),
                    mask_arr[s][np.newaxis].astype(np.int64)
                ))
        
        print(f"Total slices: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        ct, mask = self.samples[idx]
        ct_t = torch.tensor(ct)
        mask_t = torch.tensor(mask)
        
        if self.augment:
            if torch.rand(1) > 0.5:
                ct_t = torch.flip(ct_t, dims=[2])
                mask_t = torch.flip(mask_t, dims=[2])
            if torch.rand(1) > 0.5:
                ct_t = torch.flip(ct_t, dims=[1])
                mask_t = torch.flip(mask_t, dims=[1])
            # Random brightness
            ct_t = ct_t + torch.randn(1) * 0.05
            ct_t = torch.clamp(ct_t, 0, 1)
        
        return ct_t, mask_t

# ── Split patients ────────────────────────────────────────────────────────────
nifti_dir = r"D:\GSOC\nifti_output"
ts_dir = r"D:\GSOC\ts_output"

all_patients = ['1','10','100','101','102','103','104','105',
                '106','107','108','109','11','110','111','112',
                '113','114','115','116']

train_ids = all_patients[:14]  # 70%
val_ids   = all_patients[14:17]  # 15%
test_ids  = all_patients[17:]    # 15%

print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

train_ds = HeartDataset(train_ids, nifti_dir, ts_dir, augment=True)
val_ds   = HeartDataset(val_ids,   nifti_dir, ts_dir, augment=False)
test_ds  = HeartDataset(test_ids,  nifti_dir, ts_dir, augment=False)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=8, shuffle=False, num_workers=0)

# ── Model ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=7, factor=0.5, verbose=True
)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# ── Training ──────────────────────────────────────────────────────────────────
best_val_dice = 0
train_losses, val_dices = [], []

for epoch in range(60):
    model.train()
    epoch_loss = 0
    for ct, mask in train_loader:
        ct, mask = ct.to(device), mask.to(device)
        optimizer.zero_grad()
        pred = model(ct)
        loss = loss_fn(pred, mask)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # Validation
    model.eval()
    dice_metric.reset()
    with torch.no_grad():
        for ct, mask in val_loader:
            ct, mask = ct.to(device), mask.to(device)
            pred = model(ct)
            pred_binary = torch.argmax(pred, dim=1, keepdim=True)
            from monai.networks.utils import one_hot
            mask_onehot = one_hot(mask, num_classes=2)
            pred_onehot = one_hot(pred_binary, num_classes=2)
            dice_metric(y_pred=pred_onehot, y=mask_onehot)
    
    val_dice = dice_metric.aggregate().item()
    val_dices.append(val_dice)
    
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), r"D:\GSOC\best_model.pth")
    
    print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Val Dice: {val_dice:.4f} | Best: {best_val_dice:.4f}")
    
    scheduler.step(val_dice)  # ← after val_dice is computed

# ── Plot training curves ───────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses); ax1.set_title('Training Loss'); ax1.set_xlabel('Epoch')
ax2.plot(val_dices);    ax2.set_title('Validation Dice'); ax2.set_xlabel('Epoch')
plt.tight_layout()
plt.savefig(r"D:\GSOC\training_curves.png", dpi=150)
print(f"\nTraining complete. Best Val Dice: {best_val_dice:.4f}")