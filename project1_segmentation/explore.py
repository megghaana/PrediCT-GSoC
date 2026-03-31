import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

# Point this to one patient's DICOM folder
dicom_dir = r"D:\GSOC\ML4SCI\cocacoronarycalciumandchestcts-2\deidentified_nongated\1\1"

# Read the DICOM series
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
reader.SetFileNames(dicom_names)
image = reader.Execute()

# Convert to numpy
import numpy as np
arr = sitk.GetArrayFromImage(image)  # shape: (slices, H, W)

print(f"Shape: {arr.shape}")
print(f"HU range: {arr.min()} to {arr.max()}")
print(f"Spacing: {image.GetSpacing()}")

# Show middle slice
mid = arr.shape[0] // 2
plt.figure(figsize=(8, 8))
plt.imshow(arr[mid], cmap='gray', vmin=-1000, vmax=400)
plt.title(f"Slice {mid} of {arr.shape[0]}")
plt.colorbar()
plt.savefig("first_slice.png")
print("Saved first_slice.png")