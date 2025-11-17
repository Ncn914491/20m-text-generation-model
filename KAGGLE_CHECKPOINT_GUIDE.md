# Kaggle Checkpoint Resume Guide

## 🎯 Quick Overview

This guide shows you how to resume training your text generation model on Kaggle using manual checkpoint uploads.

---

## 📋 Complete Workflow

### First Training Session

1. **Run the notebook** - Training starts from scratch
2. **Wait for checkpoints** - Saved every 500 steps + end of each epoch
3. **Download before session ends**:
   - Go to Output tab (right side)
   - Download the entire `checkpoints` folder
   - Or download specific checkpoint files

### Resume Training (Next Session)

#### Step 1: Create Checkpoint Dataset

```
Kaggle → Your Profile → Datasets → New Dataset
```

- Click "New Dataset"
- Upload your checkpoint file (e.g., `checkpoint_epoch2_step1000.pt`)
- Name it: `text-gen-checkpoint-v1` (or any name)
- Description: "Training checkpoint for text generation model"
- Click "Create"

#### Step 2: Add Dataset to Notebook

```
Open Notebook → Add Data (top right) → Your Datasets
```

- Click "Add Data" button
- Select "Your Datasets" tab
- Find your checkpoint dataset
- Click "Add" next to it
- It will appear in `/kaggle/input/your-dataset-name/`

#### Step 3: Run Notebook

- Just run all cells
- Section 3 automatically detects your uploaded checkpoint
- Training resumes from that checkpoint
- New checkpoints save to `/kaggle/working/checkpoints/`

---

## 🔄 Checkpoint Management

### Automatic Features

✅ **Auto-save**: Every 500 steps + end of epoch  
✅ **Auto-cleanup**: Keeps only last 3 checkpoints  
✅ **Auto-detect**: Finds uploaded checkpoints automatically  
✅ **Best model**: Separately saved based on validation loss  

### What Gets Saved in Each Checkpoint

- Model weights
- Optimizer state
- Scheduler state
- Current epoch and step
- Training and validation loss
- Complete training history
- Timestamp

### Files You'll See

```
/kaggle/working/checkpoints/
├── checkpoint_epoch1_step500.pt
├── checkpoint_epoch2_step1000.pt
├── checkpoint_epoch2_step1500.pt  ← Latest (only last 3 kept)
├── best_model.pt                   ← Best validation loss
├── training_history.json           ← Metrics log
├── latest_checkpoint.txt           ← Points to latest
└── final_model/                    ← HuggingFace format
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer files
```

---

## 💡 Pro Tips

### Tip 1: Version Your Checkpoints
Create multiple dataset versions as you progress:
- `text-gen-checkpoint-v1` (epoch 1)
- `text-gen-checkpoint-v2` (epoch 2)
- `text-gen-checkpoint-v3` (epoch 3)

### Tip 2: Download Everything
Before session expires, download:
- Latest checkpoint (for resuming)
- Best model (for inference)
- Training history (for analysis)

### Tip 3: Multiple Checkpoints
If you upload multiple checkpoint files:
- The notebook uses the most recent one (by timestamp)
- Others are ignored but available if needed

### Tip 4: Check Before Running
Section 3 shows which checkpoint was found:
```
🔍 Searching for uploaded checkpoints...
  Found: checkpoint_epoch2_step1000.pt in text-gen-checkpoint-v1/
✓ Will use checkpoint: checkpoint_epoch2_step1000.pt
```

### Tip 5: Training History Preserved
Your training history carries over:
```python
Epoch 1: Train Loss=3.2145, Val Loss=3.1234, Perplexity=22.71
Epoch 2: Train Loss=2.8934, Val Loss=2.8123, Perplexity=16.67
Epoch 3: Train Loss=2.6543, Val Loss=2.6234, Perplexity=13.78  ← Continues here
```

---

## 🚨 Common Issues & Solutions

### Issue: "No checkpoint found"
**Solution**: Make sure you:
1. Added the dataset to the notebook (Add Data button)
2. The file ends with `.pt`
3. The filename contains "checkpoint" or "model"

### Issue: "Checkpoint file not found"
**Solution**: 
- The dataset might not be attached
- Check `/kaggle/input/` has your dataset folder
- Re-add the dataset if needed

### Issue: "Out of disk space"
**Solution**:
- The notebook keeps only last 3 checkpoints automatically
- Old checkpoints are deleted after each epoch
- Each checkpoint is ~40-50 MB

### Issue: "Training starts from scratch"
**Solution**:
- Verify checkpoint was detected in Section 3 output
- Check the checkpoint file isn't corrupted
- Try re-uploading the checkpoint

---

## 📊 Example Training Flow

### Session 1
```
Start: Epoch 1, Step 0
Train: 3 epochs
End: Epoch 3, Step 4500
Download: checkpoint_epoch3_step4500.pt
```

### Session 2
```
Upload: checkpoint_epoch3_step4500.pt as dataset
Resume: Epoch 3, Step 4500
Train: 2 more epochs
End: Epoch 5, Step 7500
Download: checkpoint_epoch5_step7500.pt
```

### Session 3
```
Upload: checkpoint_epoch5_step7500.pt as dataset
Resume: Epoch 5, Step 7500
Train: Final epochs
Complete: Full training done!
```

---

## 🎓 Training Configuration

| Parameter | Value |
|-----------|-------|
| Model Size | ~10M parameters |
| Architecture | GPT-2 style transformer |
| Dataset | WikiText-103 |
| Batch Size | 16 |
| Learning Rate | 5e-4 |
| Gradient Accumulation | 4 steps |
| Max Sequence Length | 512 tokens |
| Checkpoint Frequency | Every 500 steps |
| Checkpoints Kept | Last 3 |

---

## ✅ Checklist

### Before Starting New Session
- [ ] Downloaded checkpoint from previous session
- [ ] Created/updated Kaggle dataset with checkpoint
- [ ] Added dataset to notebook
- [ ] Verified dataset appears in "Data" tab

### During Training
- [ ] Confirmed checkpoint was detected (Section 3 output)
- [ ] Training resumed from correct epoch/step
- [ ] Monitoring loss values
- [ ] Checkpoints being saved regularly

### Before Session Ends
- [ ] Downloaded latest checkpoint
- [ ] Downloaded best model
- [ ] Downloaded training history
- [ ] Noted current epoch/step for records

---

## 🔗 Quick Links

- **Kaggle Datasets**: https://www.kaggle.com/datasets
- **Your Datasets**: https://www.kaggle.com/[username]/datasets
- **Notebook Output**: Click "Output" tab on right side

---

## 📞 Need Help?

If you encounter issues:
1. Check the notebook output in Section 3
2. Verify your dataset is attached (Data tab)
3. Ensure checkpoint file is valid (not corrupted)
4. Try re-uploading the checkpoint
5. Check file permissions on the dataset

---

**Happy Training! 🚀**
