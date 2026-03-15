# How to Select Which Dataset to Train On

## 🎯 Quick Answer

Open `train_cnn_window_split.ipynb` and look for **Section 2: Configuration**.

You'll see this:

```python
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ 🎯 STEP 1: SELECT WHICH DATASET TO TRAIN ON                            │
# │                                                                         │
# │ Uncomment ONE of the following lines:                                  │
# └─────────────────────────────────────────────────────────────────────────┘

# Option 1: Train on 1.0s window dataset (created Jan 16, 2026)
# DATASET_DIR = Path('datasets/dataset_w1.0s_h0.25s_20260116')
# WINDOW_LENGTH = 1.0

# Option 2: Train on 1.5s window dataset (created Mar 14, 2026)
DATASET_DIR = Path('datasets/dataset_w1.5s_h0.25s_20260314')
WINDOW_LENGTH = 1.5

# Option 3: Train on 2.0s window dataset (if you create one)
# DATASET_DIR = Path('datasets/dataset_w2.0s_h0.25s_YYYYMMDD')
# WINDOW_LENGTH = 2.0
```

**Just uncomment the dataset you want to use!**

---

## 📋 Step-by-Step Instructions

### To Train on 1.0s Windows:

1. Open `train_cnn_window_split.ipynb`
2. Go to **Section 2: Configuration**
3. Find these lines:
   ```python
   # Option 1: Train on 1.0s window dataset (created Jan 16, 2026)
   # DATASET_DIR = Path('datasets/dataset_w1.0s_h0.25s_20260116')
   # WINDOW_LENGTH = 1.0
   ```
4. **Remove the `#`** from the two lines:
   ```python
   # Option 1: Train on 1.0s window dataset (created Jan 16, 2026)
   DATASET_DIR = Path('datasets/dataset_w1.0s_h0.25s_20260116')
   WINDOW_LENGTH = 1.0
   ```
5. **Add `#`** to the currently active option (Option 2):
   ```python
   # Option 2: Train on 1.5s window dataset (created Mar 14, 2026)
   # DATASET_DIR = Path('datasets/dataset_w1.5s_h0.25s_20260314')
   # WINDOW_LENGTH = 1.5
   ```
6. Run the cell
7. You should see: `✓ Using dataset: datasets/dataset_w1.0s_h0.25s_20260116`

### To Train on 1.5s Windows:

1. Open `train_cnn_window_split.ipynb`
2. Go to **Section 2: Configuration**
3. Make sure these lines are **NOT** commented (no `#`):
   ```python
   # Option 2: Train on 1.5s window dataset (created Mar 14, 2026)
   DATASET_DIR = Path('datasets/dataset_w1.5s_h0.25s_20260314')
   WINDOW_LENGTH = 1.5
   ```
4. Make sure other options **ARE** commented (have `#`):
   ```python
   # Option 1: Train on 1.0s window dataset (created Jan 16, 2026)
   # DATASET_DIR = Path('datasets/dataset_w1.0s_h0.25s_20260116')
   # WINDOW_LENGTH = 1.0
   ```
5. Run the cell
6. You should see: `✓ Using dataset: datasets/dataset_w1.5s_h0.25s_20260314`

---

## ⚠️ Important Rules

### ✅ DO:
- **Uncomment exactly ONE dataset option** (remove the `#`)
- **Make sure `WINDOW_LENGTH` matches** the dataset you selected
- **Check the output** after running the cell to confirm the correct dataset is loaded

### ❌ DON'T:
- Don't uncomment multiple datasets at once (only the last one will be used)
- Don't forget to update `WINDOW_LENGTH` to match your dataset
- Don't modify the dataset path unless you created a new dataset

---

## 🔍 How to Verify You Selected the Right Dataset

After running the configuration cell, you should see:

```
======================================================================
✓ Using dataset: datasets/dataset_w1.5s_h0.25s_20260314
======================================================================

Configuration:
  Sample rate: 16000 Hz
  Window length: 1.5s
  ...
```

**Check that:**
1. The dataset path matches what you uncommented
2. The window length matches the dataset name (e.g., `w1.5s` → `1.5s`)

---

## 📊 Available Datasets

Currently, you have:

| Dataset Name | Window Size | Created | Files | Location |
|--------------|-------------|---------|-------|----------|
| `dataset_w1.0s_h0.25s_20260116` | 1.0 seconds | Jan 16, 2026 | 1,434 | `datasets/dataset_w1.0s_h0.25s_20260116/` |
| `dataset_w1.5s_h0.25s_20260314` | 1.5 seconds | Mar 14, 2026 | 1,430 | `datasets/dataset_w1.5s_h0.25s_20260314/` |

To create more datasets (e.g., 2.0s windows), use `prepare_dataset.ipynb`.

---

## 🎯 Workflow for Comparing Window Sizes

1. **Train on 1.0s dataset:**
   - Select Option 1 in configuration
   - Run all cells in `train_cnn_window_split.ipynb`
   - Model saved as: `models/glucose_alarm_cnn_w1.0s_{timestamp}.pth`

2. **Train on 1.5s dataset:**
   - Select Option 2 in configuration
   - Run all cells in `train_cnn_window_split.ipynb`
   - Model saved as: `models/glucose_alarm_cnn_w1.5s_{timestamp}.pth`

3. **Compare results:**
   - Open `model_comparison.ipynb`
   - Run all cells
   - See which window size gives better **recall** (most important for medical alarms!)

---

## ❓ Troubleshooting

### Error: "Dataset directory not found"

**Problem:** The dataset path doesn't exist.

**Solution:**
1. Check that you uncommented the correct line
2. Verify the dataset exists: `ls -la datasets/`
3. If missing, create it using `prepare_dataset.ipynb`

### Error: "Metadata file not found"

**Problem:** The dataset folder exists but is incomplete.

**Solution:**
1. Re-run `prepare_dataset.ipynb` to recreate the dataset
2. Make sure the notebook completed successfully

### Wrong window size in output

**Problem:** You uncommented the dataset but forgot to update `WINDOW_LENGTH`.

**Solution:**
1. Make sure both lines are uncommented:
   ```python
   DATASET_DIR = Path('datasets/dataset_w1.0s_h0.25s_20260116')
   WINDOW_LENGTH = 1.0  # ← Don't forget this!
   ```

---

## 💡 Pro Tip

The model name automatically includes the window size, so you can easily identify which dataset was used:

- `glucose_alarm_cnn_w1.0s_20260315_120000.pth` → Trained on 1.0s dataset
- `glucose_alarm_cnn_w1.5s_20260315_130000.pth` → Trained on 1.5s dataset

This makes it easy to compare models later!

