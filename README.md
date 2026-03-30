# PrediCT GSoC 2026 — Heart Segmentation (Project 1)

## Overview
A lightweight U-Net pipeline for automated heart segmentation in non-contrast 
cardiac CT scans, trained on the Stanford COCA dataset. Achieves **29.8x faster 
inference** than TotalSegmentator with a mean Dice score of 0.665.

## Results
| Metric | Value |
|--------|-------|
| Mean Dice (test set) | 0.665 ± 0.053 |
| Inference time (ours) | 2.5s per scan |
| TotalSegmentator time | 75s per scan |
| Speedup | 29.8x |

## Pipeline
1. `explore.py` — Load and visualize DICOM CT scans
2. `convert.py` — Convert DICOM to NIfTI format
3. `batch_process.py` — Batch convert + run TotalSegmentator for ground truth masks
4. `train_unet.py` — Train 2D U-Net with Dice loss
5. `evaluation.ipynb` — Evaluate model, generate visualizations

## Setup
```bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

## Dataset
Stanford COCA dataset. Follow instructions at KatyEB/PrediCT to download.

## Model
2D U-Net with residual units trained on 14 patients, validated on 3, tested on 3.
Ground truth masks generated using TotalSegmentator.