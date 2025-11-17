# Quick Start Guide - Kaggle Training

## 🚀 5-Minute Setup

### Step 1: Upload to Kaggle
1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **New Notebook**
3. **File** → **Import Notebook**
4. Upload `text_generation_kaggle_production.ipynb`

### Step 2: Enable GPU
- Right sidebar → **Settings**
- **Accelerator**: GPU T4 ✅
- **Internet**: ON ✅
- Click **Save**

### Step 3: Run
- Click **Run All** button
- Wait ~6-9 hours for training to complete

### Step 4: Download Results
- Go to **Output** tab
- Download:
  - `best_model.pt` (best checkpoint)
  - `final_model/` (HuggingFace format)
  - `training_history.json` (metrics)

## 📊 What You Get

- **10M parameter GPT-2 model** trained on WikiText-103
- **Smart checkpointing** (keeps 4 most recent)
- **Best model** saved automatically
- **Training history** with loss and perplexity

## ⚙️ Key Settings

```python
Batch Size: 8
Effective Batch Size: 64 (with gradient accumulation)
Learning Rate: 5e-4
Epochs: 3
Checkpoints: Every 1000 steps
Max Checkpoints Kept: 4
```

## 🔧 Common Adjustments

### Train Longer
```python
CONFIG = {
    'epochs': 5,  # Change from 3 to 5
}
```

### Save More Frequently
```python
CONFIG = {
    'save_steps': 500,  # Change from 1000 to 500
}
```

### Keep More Checkpoints
```python
CONFIG = {
    'max_checkpoints': 6,  # Change from 4 to 6
}
```

### Reduce Memory Usage
```python
CONFIG = {
    'batch_size': 4,      # Reduce from 8
    'max_length': 256,    # Reduce from 512
}
```

## 🎯 Resume Training

1. Download checkpoint from previous run
2. Upload to Kaggle Datasets
3. Add dataset to notebook
4. Update config:
```python
CONFIG = {
    'resume_from_checkpoint': '/kaggle/input/my-checkpoint/checkpoint.pt',
}
```

## 📈 Monitor Progress

Watch for:
- **Train Loss**: Should decrease over time
- **Val Loss**: Should decrease (if increasing, model is overfitting)
- **Perplexity**: Lower is better (good: <30, excellent: <20)

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| No GPU detected | Enable GPU in settings |
| Out of memory | Reduce batch_size to 4 |
| Dataset won't load | Enable internet in settings |
| Training too slow | Check GPU is enabled |
| Session expired | Download checkpoints, resume later |

## 💡 Pro Tips

1. **Enable Persistence** in settings for longer sessions
2. **Monitor GPU memory** in progress bars
3. **Download checkpoints periodically** during training
4. **Test generation** after each epoch
5. **Keep best_model.pt** - it's your safety net

## 📝 Output Files

```
/kaggle/working/
├── best_model.pt              # ⭐ Best checkpoint
├── checkpoints/               # 4 most recent
│   ├── checkpoint_epoch3_step6500.pt
│   ├── checkpoint_epoch3_step5500.pt
│   ├── checkpoint_epoch3_step4500.pt
│   └── checkpoint_epoch3_step3500.pt
├── training_history.json      # Metrics
└── final_model/               # HuggingFace format
    ├── pytorch_model.bin
    ├── config.json
    └── tokenizer files
```

## 🎓 Next Steps

After training:
1. Download all files from Output tab
2. Test the model locally
3. Fine-tune on your own dataset
4. Deploy for inference

## 📚 Full Documentation

See `KAGGLE_PRODUCTION_GUIDE.md` for complete details.

---

**Training Time**: ~6-9 hours on T4 GPU
**Model Size**: ~40 MB
**Dataset**: WikiText-103 (103M tokens)
**Parameters**: ~10 million
