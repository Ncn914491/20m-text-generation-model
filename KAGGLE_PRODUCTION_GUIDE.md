# Kaggle Production Training Guide

## Overview

This guide explains how to use the production-ready Kaggle notebook (`text_generation_kaggle_production.ipynb`) to train a 10M parameter GPT-2 style text generation model with smart checkpoint management.

## Key Features

✅ **Smart Checkpointing**: Automatically keeps only the 4 most recent checkpoints to save space
✅ **Error Handling**: Robust error handling for OOM errors and interruptions
✅ **Resume Training**: Can resume from any checkpoint
✅ **Memory Optimized**: Configured for Kaggle's GPU memory limits
✅ **Progress Tracking**: Detailed logging and progress bars
✅ **Production Ready**: Includes emergency checkpoint saving on interruption

## Quick Start

### 1. Upload to Kaggle

1. Go to [Kaggle.com](https://www.kaggle.com)
2. Navigate to **Code** → **New Notebook**
3. Click **File** → **Import Notebook**
4. Upload `text_generation_kaggle_production.ipynb`

### 2. Configure Notebook Settings

In the Kaggle notebook settings (right sidebar):
- **Accelerator**: GPU T4 or P100 (required)
- **Internet**: ON (required for downloading datasets)
- **Persistence**: ON (optional, for longer sessions)

### 3. Run the Notebook

Simply click **Run All** or run cells sequentially. The notebook will:
- Check GPU availability
- Load and tokenize the WikiText-103 dataset
- Train the model for 3 epochs
- Save checkpoints every 1000 steps
- Keep only the 4 most recent checkpoints
- Save the best model based on validation loss

## Configuration

### Model Architecture

```python
CONFIG = {
    'vocab_size': 50257,      # GPT-2 vocabulary
    'n_positions': 512,       # Max sequence length
    'n_embd': 256,           # Embedding dimension
    'n_layer': 8,            # Number of transformer layers
    'n_head': 8,             # Number of attention heads
    'n_inner': 1024,         # FFN inner dimension
}
```

**Total Parameters**: ~10 million

### Training Hyperparameters

```python
CONFIG = {
    'batch_size': 8,                    # Per-device batch size
    'gradient_accumulation_steps': 8,   # Effective batch size = 64
    'learning_rate': 5e-4,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'epochs': 3,
    'warmup_steps': 500,
    'max_length': 512,
}
```

### Checkpoint Management

```python
CONFIG = {
    'save_steps': 1000,          # Save checkpoint every N steps
    'eval_steps': 500,           # Evaluate every N steps
    'max_checkpoints': 4,        # Keep only 4 most recent
    'checkpoint_dir': '/kaggle/working/checkpoints',
}
```

## Checkpoint Management

### How It Works

The notebook automatically manages checkpoints to save disk space:

1. **Saves checkpoint every 1000 steps** during training
2. **Keeps only the 4 most recent checkpoints**
3. **Automatically deletes older checkpoints**
4. **Always keeps the best model** (separate file)

### Checkpoint Files

```
/kaggle/working/
├── best_model.pt                    # Best model (never deleted)
├── checkpoints/
│   ├── checkpoint_epoch3_step6500.pt  # Most recent
│   ├── checkpoint_epoch3_step5500.pt
│   ├── checkpoint_epoch3_step4500.pt
│   └── checkpoint_epoch3_step3500.pt  # 4th most recent
└── training_history.json
```

### Resume Training

To resume from a checkpoint, modify the configuration cell:

```python
CONFIG = {
    # ... other settings ...
    'resume_from_checkpoint': '/kaggle/input/your-checkpoint/checkpoint_epoch1_step6500.pt',
}
```

Then add the checkpoint as a dataset:
1. Upload your checkpoint to Kaggle Datasets
2. In the notebook, click **Add Data** → Select your dataset
3. Update the path in CONFIG

## Memory Optimization

### Current Settings (for T4 GPU - 16GB)

- Batch size: 8
- Gradient accumulation: 8 steps
- Effective batch size: 64
- Sequence length: 512

### If You Get OOM Errors

The notebook handles OOM errors automatically, but if they persist:

```python
CONFIG = {
    'batch_size': 4,                    # Reduce to 4
    'gradient_accumulation_steps': 16,  # Increase to 16
    'max_length': 256,                  # Reduce sequence length
}
```

### For P100 GPU (16GB)

Same settings work well.

### For V100 GPU (32GB)

You can increase batch size:

```python
CONFIG = {
    'batch_size': 16,
    'gradient_accumulation_steps': 4,
}
```

## Training Time Estimates

### On Kaggle T4 GPU

- **Per Epoch**: ~2-3 hours
- **Full Training (3 epochs)**: ~6-9 hours
- **Per 1000 steps**: ~15-20 minutes

### Tips for Long Training

1. **Enable Persistence**: In notebook settings
2. **Save Frequently**: Checkpoints every 1000 steps (default)
3. **Monitor Progress**: Check the progress bars
4. **Use Best Model**: Always saved separately

## Output Files

After training completes, you'll have:

### 1. Best Model (`best_model.pt`)
- Complete checkpoint with lowest validation loss
- Includes model, optimizer, and scheduler states
- Can be used to resume training

### 2. Checkpoints (`checkpoints/`)
- 4 most recent training checkpoints
- Saved every 1000 steps
- Older checkpoints automatically deleted

### 3. Training History (`training_history.json`)
```json
[
  {
    "epoch": 1,
    "train_loss": 3.2456,
    "val_loss": 3.1234,
    "val_perplexity": 22.71,
    "timestamp": "2024-11-17T10:30:00"
  }
]
```

### 4. Final Model (`final_model/`)
- HuggingFace format
- Ready to use with `transformers` library
- Includes tokenizer

## Using the Trained Model

### Load in Python

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model
model = GPT2LMHeadModel.from_pretrained('/kaggle/working/final_model')
tokenizer = GPT2Tokenizer.from_pretrained('/kaggle/working/final_model')

# Generate text
input_ids = tokenizer.encode("The future of AI", return_tensors='pt')
output = model.generate(input_ids, max_length=100)
text = tokenizer.decode(output[0])
print(text)
```

### Load from Checkpoint

```python
import torch

# Load checkpoint
checkpoint = torch.load('/kaggle/working/best_model.pt')

# Load into model
model.load_state_dict(checkpoint['model_state_dict'])

# Check training info
print(f"Epoch: {checkpoint['epoch']}")
print(f"Train Loss: {checkpoint['train_loss']:.4f}")
print(f"Val Loss: {checkpoint['val_loss']:.4f}")
```

## Troubleshooting

### Problem: "No GPU detected"

**Solution**: Enable GPU in notebook settings
- Click **Settings** (right sidebar)
- **Accelerator** → Select **GPU T4** or **GPU P100**
- Click **Save**

### Problem: "Out of memory" errors

**Solution**: Reduce batch size or sequence length
```python
CONFIG = {
    'batch_size': 4,
    'max_length': 256,
}
```

### Problem: Training interrupted

**Solution**: The notebook saves an emergency checkpoint
- Look for `emergency_checkpoint.pt` in `/kaggle/working/`
- Resume training using this checkpoint

### Problem: Dataset download fails

**Solution**: Enable internet access
- **Settings** → **Internet** → **ON**
- Restart the notebook

### Problem: Checkpoints taking too much space

**Solution**: Reduce max_checkpoints
```python
CONFIG = {
    'max_checkpoints': 2,  # Keep only 2 checkpoints
}
```

## Advanced Usage

### Custom Dataset

Replace the dataset loading cell:

```python
# Instead of WikiText
dataset = load_dataset('your-dataset-name')
```

### Different Model Size

Adjust the architecture:

```python
# Smaller model (~5M parameters)
CONFIG = {
    'n_embd': 192,
    'n_layer': 6,
    'n_head': 6,
    'n_inner': 768,
}

# Larger model (~20M parameters)
CONFIG = {
    'n_embd': 384,
    'n_layer': 12,
    'n_head': 12,
    'n_inner': 1536,
}
```

### Mixed Precision Training

Add to the training loop:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop
with autocast():
    outputs = model(...)
    loss = outputs.loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Best Practices

1. **Always enable GPU** - Training on CPU is impractical
2. **Monitor memory usage** - Check GPU memory in progress bars
3. **Save frequently** - Default 1000 steps is good
4. **Keep best model** - Always saved separately
5. **Check validation loss** - Monitor for overfitting
6. **Use checkpoints** - Resume if interrupted
7. **Test generation** - Verify model quality during training

## FAQ

**Q: How long does training take?**
A: ~6-9 hours for 3 epochs on T4 GPU

**Q: Can I train for more epochs?**
A: Yes, change `'epochs': 3` to any number

**Q: What if Kaggle session expires?**
A: Download checkpoints and resume in a new session

**Q: Can I use this for other languages?**
A: Yes, just change the dataset and tokenizer

**Q: How do I reduce training time?**
A: Train for fewer epochs or use a smaller model

**Q: Can I train on multiple GPUs?**
A: Kaggle provides single GPU, but code can be adapted for multi-GPU

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review Kaggle's GPU documentation
3. Check PyTorch and Transformers documentation

## License

This notebook is provided as-is for educational and research purposes.
