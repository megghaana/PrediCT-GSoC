# PrediCT GSoC 2026

This repository contains two related cardiac CT projects:

1. `project1_segmentation`
Heart segmentation from non-contrast cardiac CT using a lightweight 2D U-Net.

2. `project2_radionomics`
Radiomics feature extraction and Agatston-score analysis from gated cardiac CT.

## Repository Structure

```text
d:\GSOC
├── project1_results
│   ├── comparison_chart.png
│   ├── evaluation_results.png
│   ├── first_slice.png
│   ├── mask_overlay.png
│   └── training_curves.png
├── project1_segmentation
│   ├── batch_process.py
│   ├── best_model.pth
│   ├── convert.py
│   ├── evaluation.ipynb
│   ├── explore.py
│   ├── train_unet.py
│   └── visualise_mask.py
├── project2_radionomics
│   ├── project2_radiomics.ipynb
│   └── radiomics_features.csv
├── project2_results
│   ├── agatston_distribution.png
│   ├── correlation_analysis.png
│   └── tsne_visualization.png
├── requirements.txt
└── README.md
```

## Project 1: Heart Segmentation

### Overview
This project builds a lightweight U-Net pipeline for automated heart segmentation in non-contrast cardiac CT scans, using the Stanford COCA dataset and TotalSegmentator masks as supervision.

### Main Files
- `project1_segmentation/explore.py`: inspect and visualize CT scans
- `project1_segmentation/convert.py`: convert DICOM scans to NIfTI
- `project1_segmentation/batch_process.py`: batch conversion and TotalSegmentator mask generation
- `project1_segmentation/train_unet.py`: train the 2D U-Net model
- `project1_segmentation/evaluation.ipynb`: evaluate predictions and generate plots
- `project1_segmentation/visualise_mask.py`: visualize CT and mask overlays
- `project1_segmentation/best_model.pth`: trained segmentation model

### Outputs
- `project1_results/first_slice.png`
- `project1_results/mask_overlay.png`
- `project1_results/training_curves.png`
- `project1_results/evaluation_results.png`
- `project1_results/comparison_chart.png`

### Summary Results
| Metric | Value |
|---|---|
| Mean Dice (test set) | 0.665 +- 0.053 |
| Inference time (ours) | 2.5 s/scan |
| TotalSegmentator time | 75 s/scan |
| Speedup | 29.8x |

## Project 2: Radiomics and Calcium Analysis

### Overview
This project extracts radiomics features from calcium annotations in gated cardiac CT, computes Agatston scores, and analyzes feature relationships with calcium burden.

### Main Files
- `project2_radionomics/project2_radiomics.ipynb`: end-to-end radiomics workflow
- `project2_radionomics/radiomics_features.csv`: exported radiomics feature table

### Outputs
- `project2_results/agatston_distribution.png`
- `project2_results/correlation_analysis.png`
- `project2_results/tsne_visualization.png`

### Radiomics Table
The CSV includes:
- PyRadiomics shape features
- first-order intensity features
- GLCM texture features
- GLSZM texture features
- GLRLM texture features
- `agatston_score`
- `agatston_category`
- `n_lesions`

Features were extracted for 19 patients. One patient was skipped because the corresponding NIfTI scan was missing.

## Setup

```bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

## Dataset

The work uses the Stanford COCA dataset and related gated CT calcium annotations. Download and organize the dataset before running the pipelines.

## Notes

- The folder name is currently `project2_radionomics` in this repository and is kept as-is to match the existing workspace.
- Virtual environments such as `train_env`, `train_env2`, and `ts_env` are local development environments and are not core project outputs.
