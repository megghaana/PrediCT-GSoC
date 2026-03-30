import SimpleITK as sitk
import os
import subprocess

nongated_root = r"D:\GSOC\ML4SCI\cocacoronarycalciumandchestcts-2\deidentified_nongated"
nifti_dir = r"D:\GSOC\nifti_output"
ts_dir = r"D:\GSOC\ts_output"

os.makedirs(nifti_dir, exist_ok=True)
os.makedirs(ts_dir, exist_ok=True)

patients = sorted(os.listdir(nongated_root))[:20]
print(f"Processing {len(patients)} patients")

for patient_id in patients:
    patient_path = os.path.join(nongated_root, patient_id)
    subfolders = os.listdir(patient_path)
    dicom_dir = os.path.join(patient_path, subfolders[0])
    
    nifti_path = os.path.join(nifti_dir, f"patient_{patient_id}.nii.gz")
    ts_output = os.path.join(ts_dir, f"patient_{patient_id}")

    # Convert DICOM to NIfTI
    if not os.path.exists(nifti_path):
        print(f"[{patient_id}] Converting DICOM...")
        try:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            sitk.WriteImage(image, nifti_path)
            print(f"[{patient_id}] NIfTI saved")
        except Exception as e:
            print(f"[{patient_id}] Failed: {e}")
            continue
    else:
        print(f"[{patient_id}] NIfTI exists, skipping")

    # Run TotalSegmentator using ts_env
    if not os.path.exists(ts_output):
        print(f"[{patient_id}] Running TotalSegmentator...")
        cmd = [
            r"D:\GSOC\ts_env\Scripts\TotalSegmentator.exe",
            "-i", nifti_path,
            "-o", ts_output,
            "--fast"
        ]
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode == 0:
            print(f"[{patient_id}] Done")
        else:
            print(f"[{patient_id}] Failed")
    else:
        print(f"[{patient_id}] TS exists, skipping")

print("\nAll done!")