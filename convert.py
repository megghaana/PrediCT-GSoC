import SimpleITK as sitk
import os

def dicom_to_nifti(dicom_dir, output_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, output_path)
    print(f"Saved: {output_path}")

# Convert patient 1
dicom_dir = r"D:\GSOC\ML4SCI\cocacoronarycalciumandchestcts-2\deidentified_nongated\1\1"
output_path = r"D:\GSOC\nifti_output\patient_001.nii.gz"

os.makedirs(r"D:\GSOC\nifti_output", exist_ok=True)
dicom_to_nifti(dicom_dir, output_path)