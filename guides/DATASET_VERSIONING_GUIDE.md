# Dataset Versioning Guide

## Overview

The dataset preparation pipeline now supports creating **multiple versioned datasets** with different parameters. Each dataset is stored in its own uniquely-named folder, allowing you to experiment with different window sizes, hop lengths, and other parameters without overwriting previous datasets.

## Dataset Naming Convention

Datasets are automatically named using this format:
```
dataset_w{WINDOW_LENGTH}s_h{HOP_LENGTH}s_{DATE}
```

**Examples:**
- `dataset_w1.0s_h0.25s_20260314` - 1.0s window, 0.25s hop, created March 14, 2026
- `dataset_w1.5s_h0.25s_20260314` - 1.5s window, 0.25s hop, created March 14, 2026
- `dataset_w2.0s_h0.5s_20260314` - 2.0s window, 0.5s hop, created March 14, 2026

## Directory Structure

```
Alarm ML classification/
├── datasets/                          # Parent folder for all datasets
│   ├── dataset_w1.0s_h0.25s_20260314/
│   │   ├── train/                     # Training windows
│   │   ├── val/                       # Validation windows
│   │   └── dataset_metadata.csv       # Metadata for this dataset
│   ├── dataset_w1.5s_h0.25s_20260314/
│   │   ├── train/
│   │   ├── val/
│   │   └── dataset_metadata.csv
│   └── dataset_w2.0s_h0.25s_20260314/
│       ├── train/
│       ├── val/
│       └── dataset_metadata.csv
├── models/                            # All trained models
│   ├── glucose_alarm_cnn_w1.0s_20260314_143022.pth
│   ├── glucose_alarm_cnn_w1.0s_20260314_143022_metadata.json
│   ├── glucose_alarm_cnn_w1.5s_20260314_150000.pth
│   └── glucose_alarm_cnn_w1.5s_20260314_150000_metadata.json
└── sessions/                          # Original audio recordings
```

## Workflow: Creating and Comparing Multiple Datasets

### Step 1: Create Dataset with 1.0s Windows

1. Open `prepare_dataset.ipynb`
2. Set parameters:
   ```python
   WINDOW_LENGTH = 1.0  # seconds
   HOP_LENGTH = 0.25    # seconds
   ```
3. Run all cells
4. Dataset created at: `datasets/dataset_w1.0s_h0.25s_20260314/`

### Step 2: Create Dataset with 1.5s Windows

1. In the same notebook, change:
   ```python
   WINDOW_LENGTH = 1.5  # seconds
   HOP_LENGTH = 0.25    # seconds
   ```
2. Run all cells again
3. Dataset created at: `datasets/dataset_w1.5s_h0.25s_20260314/`

### Step 3: Create Dataset with 2.0s Windows

1. Change parameters:
   ```python
   WINDOW_LENGTH = 2.0  # seconds
   HOP_LENGTH = 0.25    # seconds
   ```
2. Run all cells
3. Dataset created at: `datasets/dataset_w2.0s_h0.25s_20260314/`

### Step 4: Train Models on Each Dataset

1. Open `train_cnn_window_split.ipynb`
2. For the **1.0s dataset**:
   ```python
   DATASET_DIR = Path('datasets/dataset_w1.0s_h0.25s_20260314')
   ```
3. Run all cells → Model saved with metadata

4. For the **1.5s dataset**:
   ```python
   DATASET_DIR = Path('datasets/dataset_w1.5s_h0.25s_20260314')
   ```
5. Run all cells → Model saved with metadata

6. For the **2.0s dataset**:
   ```python
   DATASET_DIR = Path('datasets/dataset_w2.0s_h0.25s_20260314')
   ```
7. Run all cells → Model saved with metadata

### Step 5: Compare Results

1. Open `model_comparison.ipynb`
2. Run all cells
3. Review the comparison table and visualizations
4. Identify which window size gives the best **recall** (most important for medical alarms!)

## Key Benefits

✅ **No Data Loss** - Old datasets are never overwritten  
✅ **Easy Comparison** - All datasets stored side-by-side  
✅ **Reproducibility** - Date stamp shows when dataset was created  
✅ **Clear Naming** - Parameters visible in folder name  
✅ **Systematic Experiments** - Test multiple configurations easily

## Important Notes

- **Disk Space**: Each dataset takes ~50-100 MB depending on window size
- **Old Dataset**: Your original `dataset/` folder is still there and can be deleted once you've migrated to the new system
- **Model Metadata**: Each trained model automatically records which dataset it was trained on
- **Recommended Window Sizes**: Start with 1.0s, 1.5s, and 2.0s to find the optimal balance

## Troubleshooting

**Error: "Dataset directory not found"**
- Make sure you've run `prepare_dataset.ipynb` first to create the dataset
- Check that the `DATASET_DIR` path in `train_cnn_window_split.ipynb` matches an existing dataset folder

**Error: "Metadata file not found"**
- The dataset preparation didn't complete successfully
- Re-run `prepare_dataset.ipynb` with the desired parameters

## Next Steps

1. Create 3 datasets (1.0s, 1.5s, 2.0s windows)
2. Train a model on each dataset
3. Compare results in `model_comparison.ipynb`
4. Choose the best model based on **recall** performance
5. Use the best model in `live_inference.ipynb`

