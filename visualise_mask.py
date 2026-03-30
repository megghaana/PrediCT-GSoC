import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# Load CT scan
ct = sitk.ReadImage(r"D:\GSOC\nifti_output\patient_100.nii.gz")
ct_arr = sitk.GetArrayFromImage(ct)

# Load heart mask
mask = sitk.ReadImage(r"D:\GSOC\ts_output\patient_100\heart.nii.gz")
mask_arr = sitk.GetArrayFromImage(mask)

print(f"CT shape: {ct_arr.shape}")
print(f"Mask shape: {mask_arr.shape}")
print(f"Heart voxels: {mask_arr.sum()}")

# Find middle slice of heart
heart_slices = np.where(mask_arr.sum(axis=(1,2)) > 0)[0]
mid_slice = heart_slices[len(heart_slices)//2]

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# CT only
axes[0].imshow(ct_arr[mid_slice], cmap='gray', vmin=-1000, vmax=400)
axes[0].set_title('CT Scan')

# Mask only
axes[1].imshow(mask_arr[mid_slice], cmap='hot')
axes[1].set_title('Heart Mask')

# Overlay
axes[2].imshow(ct_arr[mid_slice], cmap='gray', vmin=-1000, vmax=400)
axes[2].imshow(mask_arr[mid_slice], cmap='hot', alpha=0.4)
axes[2].set_title('Overlay')

plt.tight_layout()
plt.savefig('mask_overlay.png', dpi=150)
print(f"Saved mask_overlay.png (slice {mid_slice})")