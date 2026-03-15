# Model Tracking System Guide

## Overview

The training notebook now automatically tracks all experiments with comprehensive metadata, making it easy to compare different models and find the best one for your use case.

---

## 🎯 What Gets Tracked

Every time you train a model, the following information is automatically saved:

### 1. **Model Identification**
- Unique model name with timestamp
- Model file path

### 2. **Audio Parameters**
- Sample rate
- **Window length** (key parameter!)
- Number of mel bands
- FFT size
- Hop length

### 3. **Training Parameters**
- Batch size
- Number of epochs
- Learning rate
- Optimizer type
- Loss function

### 4. **Dataset Information**
- Total samples
- Train/validation split sizes
- Class distribution

### 5. **Performance Metrics**
- **Accuracy**
- **Precision**
- **Recall** (most important for medical use!)
- **F1 Score**
- Best validation loss
- Best epoch
- Training history (all epochs)

### 6. **Confusion Matrix**
- True Positives
- True Negatives
- False Positives
- False Negatives

### 7. **System Information**
- Device (CPU/GPU)
- Platform
- Python version
- PyTorch version

---

## 📁 File Structure

After training, you'll have:

```
models/
├── glucose_alarm_cnn_w1.0s_20260314_143022.pth           # Model weights
├── glucose_alarm_cnn_w1.0s_20260314_143022_metadata.json # Detailed metadata
├── glucose_alarm_cnn_w1.5s_20260314_150133.pth           # Another model
├── glucose_alarm_cnn_w1.5s_20260314_150133_metadata.json
├── training_log.jsonl                                     # Master log (all models)
├── model_comparison.csv                                   # Comparison table
└── model_comparison_report.txt                            # Summary report
```

---

## 🚀 How to Use

### Training a New Model

1. **Modify parameters** in `train_cnn_window_split.ipynb`:
   ```python
   WINDOW_LENGTH = 1.5  # Try different window sizes!
   NUM_EPOCHS = 20
   LEARNING_RATE = 0.001
   ```

2. **Run all cells** - The model will be automatically saved with:
   - Unique timestamped name
   - Complete metadata
   - Entry in master log

3. **Check the output** - You'll see:
   ```
   ✓ Model metadata saved to: models/glucose_alarm_cnn_w1.5s_20260314_143022_metadata.json
   ✓ Training log updated: models/training_log.jsonl
   ```

---

### Comparing Models

Open `model_comparison.ipynb` and run all cells to:

1. **View all models** in a comparison table
2. **Find best models** by different metrics:
   - Best accuracy
   - Best recall (catches most alarms)
   - Best F1 score (balanced)
3. **Visualize performance** vs window size
4. **Compare specific models** in detail
5. **View training curves** for any model
6. **Export reports** (CSV and text)

---

## 📊 Key Metrics Explained

### For Medical Use (Glucose Alarm Detection)

**Priority Order**:

1. **Recall** (Sensitivity) - Most Important! 🚨
   - Percentage of actual alarms that were detected
   - **Higher is better** (you want to catch ALL alarms)
   - Example: 0.90 = catches 90% of alarms, misses 10%

2. **Precision** - Important
   - Percentage of alarm predictions that were correct
   - Lower precision = more false alarms (annoying but safe)
   - Example: 0.85 = 85% of alerts are real, 15% are false

3. **F1 Score** - Balanced metric
   - Harmonic mean of precision and recall
   - Good for overall comparison

4. **Accuracy** - Less important
   - Overall correctness
   - Can be misleading with imbalanced data

### False Negative Rate (Critical!)

- **False Negative Rate = Missed Alarms / Total Alarms**
- This is the **most dangerous** metric
- Example: 0.10 = 10% of alarms are missed
- **Goal**: Minimize this! Aim for < 5%

---

## 🎯 Experiment Workflow

### Example: Finding the Best Window Size

1. **Train with 1.0s windows**:
   ```python
   WINDOW_LENGTH = 1.0
   ```
   Run training → Check recall

2. **Train with 1.5s windows**:
   ```python
   WINDOW_LENGTH = 1.5
   ```
   Run training → Check recall

3. **Train with 2.0s windows**:
   ```python
   WINDOW_LENGTH = 2.0
   ```
   Run training → Check recall

4. **Compare all models**:
   - Open `model_comparison.ipynb`
   - Run all cells
   - Look at "Best Recall" section
   - Check visualization of Recall vs Window Size

5. **Choose the winner**:
   - Pick model with highest recall
   - If recall is similar, pick higher precision
   - Update `live_inference.ipynb` to use that model

---

## 📈 Reading the Comparison Report

Example output:

```
🏆 BEST MODELS BY METRIC
================================================================================

📊 Best Accuracy:
Model Name                                    Window (s)  Accuracy  Recall
glucose_alarm_cnn_w1.5s_20260314_143022      1.5         0.9123    0.9200

🎯 Best Recall (Most Important for Medical Use):
Model Name                                    Window (s)  Recall    Accuracy
glucose_alarm_cnn_w2.0s_20260314_150133      2.0         0.9350    0.9050

⚖️ Best F1 Score (Balanced):
Model Name                                    Window (s)  F1 Score  Accuracy  Recall
glucose_alarm_cnn_w1.5s_20260314_143022      1.5         0.9100    0.9123    0.9200
```

**Interpretation**:
- The 2.0s window model has the best recall (93.5%) - catches most alarms!
- The 1.5s window model has the best accuracy (91.23%)
- For medical use, **choose the 2.0s model** (prioritize recall)

---

## 💡 Tips

1. **Always check recall first** for medical applications
2. **Compare at least 3 window sizes** (e.g., 1.0s, 1.5s, 2.0s)
3. **Keep training parameters consistent** when testing window sizes
4. **Save the comparison report** before making changes
5. **Document your choice** in the model name or notes

---

## 🔍 Viewing Individual Model Details

To see full details of a specific model:

```python
# In model_comparison.ipynb
compare_models(['glucose_alarm_cnn_w1.5s_20260314_143022'])
```

This shows:
- All parameters
- Full confusion matrix
- False negative/positive rates
- Training history

---

## ✅ Best Practices

1. **Train multiple models** with different window sizes
2. **Use model_comparison.ipynb** to find the best one
3. **Prioritize recall** over accuracy for medical use
4. **Keep the master log** (`training_log.jsonl`) for long-term tracking
5. **Export reports** before major changes

---

## 🎉 Summary

You now have a complete experiment tracking system that:
- ✅ Automatically saves all model metadata
- ✅ Tracks every parameter and metric
- ✅ Makes it easy to compare experiments
- ✅ Helps you find the best model for your use case
- ✅ Provides detailed reports and visualizations

**Next Steps**: Train models with different window sizes and use the comparison notebook to find the best one!

