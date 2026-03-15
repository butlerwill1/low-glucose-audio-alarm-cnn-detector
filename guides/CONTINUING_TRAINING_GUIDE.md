# How to Continue Training from a Previous Model

## 🎯 When to Continue Training

Continue training from a previous model when:

✅ **Best model is at the last epoch** (still improving)
✅ **Validation loss is still decreasing** (not plateaued)
✅ **No overfitting** (small gap between train and val loss)
✅ **Training was stable** (few spikes)

**Example from your recent training:**
```
Epoch [38/40] | Val Loss: 0.2185 ✅ Improving
Epoch [40/40] | Val Loss: 0.1934 ✅ Still improving! (last epoch)

✅ Perfect case for continuing!
```

---

## 📋 Step-by-Step Instructions

### **Step 1: Identify Your Best Model**

After training completes, note the model path:
```
Training complete!
Best model from epoch 40 with val_loss: 0.1934
Model saved to: models/glucose_alarm_cnn_w1.5s_20260315_092736.pth
```

Copy this path: `models/glucose_alarm_cnn_w1.5s_20260315_092736.pth`

---

### **Step 2: Update Configuration**

Open `train_cnn_window_split.ipynb` and go to **Section 2: Configuration**.

Find this section:
```python
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ 🔄 STEP 2: CONTINUE FROM PREVIOUS MODEL? (Optional)                    │
# └─────────────────────────────────────────────────────────────────────────┘

# Option A: Start from scratch (default)
LOAD_MODEL_PATH = None

# Option B: Continue from previous model (uncomment and update path)
# LOAD_MODEL_PATH = 'models/glucose_alarm_cnn_w1.5s_20260315_092736.pth'
```

**Change it to:**
```python
# Option A: Start from scratch (default)
# LOAD_MODEL_PATH = None  ← Comment this out

# Option B: Continue from previous model (uncomment and update path)
LOAD_MODEL_PATH = 'models/glucose_alarm_cnn_w1.5s_20260315_092736.pth'  ← Uncomment this
```

---

### **Step 3: Configure Training Parameters**

**Option A: Keep Same Hyperparameters (Recommended)**
```python
LEARNING_RATE = 0.001  # Same as before
BATCH_SIZE = 32        # Same as before
NUM_EPOCHS = 20        # Train for 20 more epochs (total: 60)
```

**Option B: Reduce Learning Rate for Fine-Tuning**
```python
LEARNING_RATE = 0.0005  # Half of previous (0.001)
BATCH_SIZE = 32
NUM_EPOCHS = 20
```

**Option C: Use Learning Rate Scheduler (Best!)**
```python
LEARNING_RATE = 0.001          # Start with same LR
BATCH_SIZE = 32
NUM_EPOCHS = 30

# Enable scheduler
USE_LR_SCHEDULER = True        # Automatically reduces LR when plateauing
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_MIN_LR = 1e-6
```

---

### **Step 4: Run Training**

Run all cells in the notebook. You'll see:

```
======================================================================
✓ Using dataset: datasets/dataset_w1.5s_h0.25s_20260314
======================================================================

🔄 Continuing training from: models/glucose_alarm_cnn_w1.5s_20260315_092736.pth

Configuration:
  Sample rate: 16000 Hz
  Window length: 1.5s
  ...
  Model name: glucose_alarm_cnn_w1.5s_20260315_143022_continued
  Save path: models/glucose_alarm_cnn_w1.5s_20260315_143022_continued.pth
======================================================================
```

**Notice:**
- ✅ New model name includes `_continued` suffix
- ✅ New timestamp (20260315_143022)
- ✅ Old model is preserved (not overwritten!)

---

### **Step 5: Monitor Training**

Watch for continued improvement:

```
Starting training...

Epoch [ 1/20] | Train Loss: 0.1923 | Val Loss: 0.1912 ✅ Improving
Epoch [ 5/20] | Train Loss: 0.1756 | Val Loss: 0.1823 ✅ Improving
Epoch [10/20] | Train Loss: 0.1645 | Val Loss: 0.1756 ✅ Improving
Epoch [15/20] | Train Loss: 0.1589 | Val Loss: 0.1723 ✅ Improving
Epoch [20/20] | Train Loss: 0.1534 | Val Loss: 0.1698 ✅ Best!

Training complete!
Best model from epoch 20 with val_loss: 0.1698
```

---

## 🔍 What Happens Behind the Scenes

### **Model Loading:**
```python
if LOAD_MODEL_PATH is not None:
    model.load_state_dict(torch.load(LOAD_MODEL_PATH))
    print("✅ Weights loaded successfully!")
```

### **New Model Name:**
```python
if LOAD_MODEL_PATH is not None:
    MODEL_NAME = f'glucose_alarm_cnn_w{WINDOW_LENGTH}s_{TIMESTAMP}_continued'
else:
    MODEL_NAME = f'glucose_alarm_cnn_w{WINDOW_LENGTH}s_{TIMESTAMP}'
```

### **Result:**
- Old model: `glucose_alarm_cnn_w1.5s_20260315_092736.pth` (preserved!)
- New model: `glucose_alarm_cnn_w1.5s_20260315_143022_continued.pth` (new file!)

---

## 📊 Expected Results

### **Scenario 1: Continued Improvement**
```
Old model (epoch 40):  Val Loss: 0.1934, Recall: ~92%
New model (epoch 60):  Val Loss: 0.1698, Recall: ~94% ✅ Better!
```

### **Scenario 2: Plateau Reached**
```
Old model (epoch 40):  Val Loss: 0.1934, Recall: ~92%
New model (epoch 60):  Val Loss: 0.1912, Recall: ~92% ← Marginal improvement
```

### **Scenario 3: Overfitting Starts**
```
Old model (epoch 40):  Train: 0.2067, Val: 0.1934 (gap: 0.013)
New model (epoch 60):  Train: 0.1234, Val: 0.2145 (gap: 0.091) ❌ Overfitting!
```

---

## ⚠️ When to Stop Continuing

**Stop if you see:**

1. **Validation loss plateaus for 10+ epochs:**
   ```
   Epoch 50: Val Loss: 0.1756
   Epoch 60: Val Loss: 0.1754  ← Barely improving
   ```

2. **Overfitting starts:**
   ```
   Train loss ↓↓↓ but Val loss ↑↑↑
   ```

3. **You hit your target:**
   ```
   Recall: 95.2% ✅ Target achieved!
   ```

---

## 🎯 Pro Tips

1. **Always keep the old model** - The `_continued` suffix ensures this
2. **Start with same hyperparameters** - Only change if you have a reason
3. **Use learning rate scheduler** - It will automatically reduce LR when needed
4. **Monitor for overfitting** - Watch the train/val gap
5. **Compare models** - Use `model_comparison.ipynb` to compare old vs new

---

## 🔄 Multiple Continuations

You can continue multiple times:

```
Training 1: glucose_alarm_cnn_w1.5s_20260315_092736.pth (epochs 1-40)
            ↓ Continue
Training 2: glucose_alarm_cnn_w1.5s_20260315_143022_continued.pth (epochs 41-60)
            ↓ Continue again
Training 3: glucose_alarm_cnn_w1.5s_20260315_165543_continued.pth (epochs 61-80)
```

Just update `LOAD_MODEL_PATH` to the most recent model each time!

---

## 📋 Quick Checklist

Before continuing training:

- [ ] Best model is at the last epoch?
- [ ] Validation loss still decreasing?
- [ ] No overfitting (small train/val gap)?
- [ ] Copied the correct model path?
- [ ] Updated `LOAD_MODEL_PATH` in configuration?
- [ ] Decided on hyperparameters (same or reduced LR)?
- [ ] Learning rate scheduler enabled?

If all ✅ → **Ready to continue!** 🚀


