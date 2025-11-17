# Kaggle Training Guide

## Overview
This guide explains how to train your model on Kaggle and resume from checkpoints using JSON format (since .pt files can't be uploaded as datasets).

## Files

1. **checkpoint_to_json.py** - Converts PyTorch .pt checkpoints to JSON format
2. **text_generation_model_kaggle.ipynb** - Kaggle notebook for training with JSON checkpoint support
3. **checkpoint_epoch1_step6500.pt** - Your existing checkpoint (local only)

## Step-by-Step Process

### Step 1: Convert Checkpoint to JSON

Run the conversion script locally:

```bash
python checkpoint_to_json.py checkpoint_epoch1_step6500.pt --output checkpoint_epoch1_step6500.json --test-load
```

This will:
- Convert your .pt checkpoint to JSON format
- Create `checkpoint_epoch1_step6500.json`
- Test loading the JSON to verify it works
- Show file size comparison

**Note:** JSON files are larger than .pt files (typically 2-3x), but they can be uploaded to Kaggle as datasets.

### Step 2: Upload JSON Checkpoint to Kaggle

1. Go to Kaggle.com
2. Navigate to "Datasets" → "New Dataset"
3. Upload `checkpoint_epoch1_step6500.json`
4. Name it something like "text-gen-checkpoint-epoch1"
5. Make it public or private as needed
6. Note the dataset path (e.g., `/kaggle/input/text-gen-checkpoint-epoch1/checkpoint_epoch1_step6500.json`)

### Step 3: Create Kaggle Notebook

1. Go to "Code" → "New Notebook"
2. Upload `text_generation_model_kaggle.ipynb`
3. Or copy-paste the cells from the notebook

### Step 4: Configure the Notebook

In the notebook's configuration cell, update:

```python
CONFIG = {
    'resume_from_json': True,  # Set to True to resume
    'json_checkpoint_path': '/kaggle/input/text-gen-checkpoint-epoch1/checkpoint_epoch1_step6500.json',  # Update path
    # ... other settings
}
```

### Step 5: Add Dataset to Notebook

1. In your Kaggle notebook, click "Add Data" on the right
2. Search for your uploaded checkpoint dataset
3. Add it to the notebook
4. Verify the path matches your CONFIG

### Step 6: Enable GPU and Run

1. In notebook settings, enable GPU (T4 or P100)
2. Enable internet access (needed for downloading datasets)
3. Run all cells
4. Training will resume from your checkpoint

### Step 7: Download Results

After training completes, download from `/kaggle/working/`:
- `best_model.json` - Best checkpoint
- `checkpoint_epoch*.json` - Epoch checkpoints
- `training_history.json` - Training metrics
- `final_model/` - HuggingFace format model

## Training from Scratch

If you want to train from scratch instead:

```python
CONFIG = {
    'resume_from_json': False,  # Set to False
    # ... other settings
}
```

## Converting JSON Back to PyTorch

To use the JSON checkpoint locally:

```python
from checkpoint_to_json import load_json_checkpoint

# Load checkpoint
checkpoint = load_json_checkpoint('checkpoint_epoch1_step6500.json')

# Load into model
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## Memory Optimization Tips

The notebook is configured for Kaggle's GPU memory:

- **Batch size**: 8 (reduced from 16)
- **Gradient accumulation**: 8 steps (effective batch size = 64)
- **Automatic cache clearing** after each epoch
- **Mixed precision** can be added if needed

## Troubleshooting

### JSON file too large
- The optimizer state takes up most space
- You can skip optimizer state if starting fresh:
  ```python
  CONFIG['resume_from_json'] = True  # Load model weights only
  # Optimizer will be reinitialized
  ```

### Out of memory
- Reduce batch_size to 4
- Increase gradient_accumulation_steps to 16
- Reduce max_length to 256

### Checkpoint not found
- Verify the dataset is added to your notebook
- Check the path in CONFIG matches the actual path
- Use `!ls /kaggle/input/` to see available datasets

## Continuous Training Workflow

1. Train on Kaggle for a few epochs
2. Download the checkpoint JSON
3. Upload as new dataset version
4. Resume training in a new notebook session
5. Repeat as needed

## File Size Comparison

Typical sizes for a 10M parameter model:
- `.pt` checkpoint: ~40-50 MB
- `.json` checkpoint: ~100-150 MB (2-3x larger)
- HuggingFace model: ~40 MB

The JSON format is less efficient but works with Kaggle's dataset system.

## Alternative: Training from Scratch

Since you have the model architecture defined, you can also:
1. Train from scratch on Kaggle (no checkpoint needed)
2. Save checkpoints as JSON during training
3. Download and continue locally or in another Kaggle session

This avoids the initial upload but means starting over.
