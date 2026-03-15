# Model Tracking System - Changes Summary

## 🎯 What Was Added

A comprehensive model tracking and comparison system to help you systematically test different parameters (especially window sizes) and find the best model.

---

## 📝 Files Modified

### 1. `train_cnn_window_split.ipynb` - Enhanced Training Notebook

**Changes**:
- ✅ **Automatic unique model naming** with timestamp
  - Old: `glucose_alarm_cnn_window_split.pth` (overwrites previous)
  - New: `glucose_alarm_cnn_w1.5s_20260314_143022.pth` (keeps all versions)

- ✅ **Comprehensive metadata saving** after each training run
  - Saves all parameters (window size, learning rate, etc.)
  - Saves all metrics (accuracy, precision, recall, F1)
  - Saves confusion matrix
  - Saves complete training history
  - Saves system information

- ✅ **Master log file** (`training_log.jsonl`)
  - Appends each training run to a single file
  - Easy to track all experiments over time

**What you'll see**:
```
Configuration:
  Sample rate: 16000 Hz
  Window length: 1.5s
  Mel bands: 64
  ...
  Model name: glucose_alarm_cnn_w1.5s_20260314_143022
  Save path: models/glucose_alarm_cnn_w1.5s_20260314_143022.pth

[Training happens...]

✓ Model metadata saved to: models/glucose_alarm_cnn_w1.5s_20260314_143022_metadata.json
✓ Training log updated: models/training_log.jsonl
```

---

## 📁 Files Created

### 2. `model_comparison.ipynb` - Model Comparison Notebook

**Features**:
- 📊 **Comparison table** showing all trained models
- 🏆 **Best model finder** by different metrics:
  - Best accuracy
  - Best recall (most important for medical use!)
  - Best F1 score
- 📈 **Visualizations**:
  - Performance vs window size scatter plots
  - Training curves for any model
- 🔍 **Detailed comparison** of specific models
- 📄 **Export reports** (CSV and text files)

**Sections**:
1. Load all model metadata
2. Create comparison table
3. Find best models
4. Visualize performance by window size
5. Compare specific models in detail
6. View training history
7. Export comparison reports

### 3. `MODEL_TRACKING_GUIDE.md` - Complete Documentation

**Contents**:
- What gets tracked
- File structure explanation
- How to use the system
- Key metrics explained (especially for medical use)
- Experiment workflow examples
- Tips and best practices

### 4. `CHANGES_SUMMARY.md` - This file!

Quick reference for what changed and how to use it.

---

## 🚀 How to Use the New System

### Step 1: Train Multiple Models

```python
# In train_cnn_window_split.ipynb

# First experiment: 1.0s windows
WINDOW_LENGTH = 1.0
# Run all cells

# Second experiment: 1.5s windows  
WINDOW_LENGTH = 1.5
# Run all cells

# Third experiment: 2.0s windows
WINDOW_LENGTH = 2.0
# Run all cells
```

Each run creates:
- `models/glucose_alarm_cnn_w1.0s_TIMESTAMP.pth`
- `models/glucose_alarm_cnn_w1.0s_TIMESTAMP_metadata.json`
- Entry in `models/training_log.jsonl`

### Step 2: Compare Models

Open `model_comparison.ipynb` and run all cells.

You'll see:
```
Found 3 trained models

MODEL COMPARISON TABLE
================================================================================
Model Name                          Window (s)  Accuracy  Recall  F1 Score  ...
glucose_alarm_cnn_w1.0s_20260314... 1.0         0.8571    0.8641  0.8581
glucose_alarm_cnn_w1.5s_20260314... 1.5         0.9123    0.9200  0.9100
glucose_alarm_cnn_w2.0s_20260314... 2.0         0.9050    0.9350  0.9150
================================================================================

🏆 BEST MODELS BY METRIC
================================================================================

🎯 Best Recall (Most Important for Medical Use):
Model Name                          Window (s)  Recall    Accuracy
glucose_alarm_cnn_w2.0s_20260314... 2.0         0.9350    0.9050
```

### Step 3: Choose the Best Model

Based on the comparison:
1. **For medical use**: Choose highest recall (catches most alarms)
2. **For general use**: Choose highest F1 score (balanced)
3. **For low false alarms**: Choose highest precision

### Step 4: Update Live Inference

```python
# In live_inference.ipynb

# Update to use your best model
MODEL_PATH = "models/glucose_alarm_cnn_w2.0s_20260314_150133.pth"
WINDOW_DURATION = 2.0  # Match the window size!
```

---

## 📊 Example Workflow: Finding Best Window Size

This addresses your original question about whether 1s windows are too short!

### Experiment Plan

1. **Hypothesis**: Longer windows (1.5-2s) will improve recall because they always contain a beep

2. **Test**:
   - Train with 1.0s windows (baseline)
   - Train with 1.5s windows
   - Train with 2.0s windows
   - Train with 3.0s windows (optional)

3. **Compare**:
   - Open `model_comparison.ipynb`
   - Look at "Best Recall" section
   - Check scatter plot of Recall vs Window Size

4. **Decide**:
   - Pick the window size with highest recall
   - If recall is similar, pick the one with higher precision
   - Avoid windows that are too long (>3s) per ChatGPT's advice

5. **Deploy**:
   - Update `live_inference.ipynb` with the best model
   - Update `WINDOW_DURATION` to match

---

## 🎯 Key Benefits

### Before (Old System)
- ❌ Models overwrite each other
- ❌ No record of parameters used
- ❌ Hard to compare experiments
- ❌ Manual tracking in notes
- ❌ Can't remember which model was best

### After (New System)
- ✅ All models saved with unique names
- ✅ Complete metadata for every experiment
- ✅ Easy comparison with one notebook
- ✅ Automatic tracking
- ✅ Clear visualization of results
- ✅ Export reports for documentation

---

## 📈 What Gets Tracked (Summary)

Every model saves:
- **Parameters**: Window size, learning rate, batch size, epochs, etc.
- **Metrics**: Accuracy, precision, recall, F1 score
- **Confusion Matrix**: TP, TN, FP, FN
- **Training History**: Loss and accuracy for every epoch
- **Dataset Info**: Number of samples, class distribution
- **System Info**: Device, Python version, PyTorch version

---

## 💡 Pro Tips

1. **Keep training parameters consistent** when testing window sizes
   - Only change `WINDOW_LENGTH`
   - Keep `NUM_EPOCHS`, `LEARNING_RATE`, etc. the same

2. **Run at least 3 experiments** before deciding
   - More data points = better decision

3. **Prioritize recall for medical use**
   - Missing an alarm is worse than a false alarm

4. **Check the confusion matrix**
   - Look at false negatives (missed alarms)
   - Goal: Minimize false negatives

5. **Save comparison reports**
   - Export CSV and text reports
   - Keep for documentation

---

## 🎉 Ready to Use!

You now have a professional experiment tracking system. 

**Next steps**:
1. ✅ Read `MODEL_TRACKING_GUIDE.md` for detailed instructions
2. ✅ Train models with different window sizes (1.0s, 1.5s, 2.0s)
3. ✅ Use `model_comparison.ipynb` to find the best one
4. ✅ Update `live_inference.ipynb` with the winner

**Questions?** Check the guide or ask!

