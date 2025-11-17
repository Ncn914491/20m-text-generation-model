# 20M Parameter Text Generation Model

A lightweight transformer-based language model with ~20 million parameters for text generation and NLP tasks, optimized for Google Colab training.

## Project Overview

This project implements a GPT-2 style transformer model with the following specifications:
- **Parameters**: ~20 million
- **Architecture**: 8-layer transformer with 8 attention heads
- **Embedding Dimension**: 256
- **Context Length**: 512 tokens
- **Training Dataset**: WikiText-103

## Features

- Efficient 20M parameter architecture
- Text generation with temperature and sampling controls
- Comprehensive benchmarking suite
- Training progress tracking
- Model checkpointing
- Performance metrics (perplexity, tokens/sec)

## Setup Instructions

### 1. Google Colab Setup

1. Open Google Colab: https://colab.research.google.com/
2. Upload `text_generation_model.ipynb`
3. Set runtime to GPU:
   - Runtime → Change runtime type → GPU (T4 recommended)
4. Run all cells sequentially

### 2. Local Setup (Optional)

```bash
pip install torch transformers datasets tokenizers accelerate evaluate rouge-score nltk sacrebleu
```

## Model Architecture

```
GPT2Config:
- vocab_size: 50,257
- n_positions: 512
- n_embd: 256
- n_layer: 8
- n_head: 8
- n_inner: 1024
```

**Estimated Parameters**: ~20,000,000

## Training Configuration

- **Batch Size**: 16
- **Learning Rate**: 5e-4
- **Epochs**: 3
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Linear warmup
- **Gradient Accumulation**: 4 steps
- **Max Gradient Norm**: 1.0

## Usage

### Training

Run the notebook cells in order. Training takes approximately 2-4 hours on a T4 GPU.

### Text Generation

```python
prompt = "The future of artificial intelligence"
generated_text = generate_text(
    prompt, 
    max_length=100, 
    temperature=0.8,
    top_k=50,
    top_p=0.95
)
print(generated_text)
```

### Loading Saved Model

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('./final_model')
tokenizer = GPT2Tokenizer.from_pretrained('./final_model')
```

## Benchmarks

Expected performance metrics:
- **Validation Perplexity**: 25-35 (after 3 epochs)
- **Inference Speed**: 5000-8000 tokens/sec (T4 GPU)
- **Model Size**: ~40 MB (FP32)
- **Training Time**: 2-4 hours (T4 GPU, 3 epochs)

## Files

- `text_generation_model.ipynb` - Main training notebook
- `text_generation_model_gdrive_checkpoints.ipynb` - Training with Google Drive checkpoint support
- `README.md` - This file
- `requirements.txt` - Python dependencies
- `best_model.pt` - Best model checkpoint (generated during training)
- `metrics_summary.json` - Training metrics (generated during training)

## Tips for Better Results

1. **Increase training epochs** (5-10) for better convergence
2. **Use larger datasets** like OpenWebText for improved quality
3. **Adjust temperature** (0.7-1.0) for generation diversity
4. **Enable wandb** for experiment tracking
5. **Save to Google Drive** to persist models across sessions

## Troubleshooting

### Out of Memory
- Reduce batch size to 8 or 4
- Increase gradient accumulation steps
- Use mixed precision training (FP16)

### Slow Training
- Ensure GPU runtime is enabled
- Check GPU utilization with `!nvidia-smi`
- Reduce sequence length to 256

### Poor Generation Quality
- Train for more epochs
- Increase model size (12-16 layers)
- Use better quality datasets
- Adjust sampling parameters

## Future Improvements

- [ ] Add mixed precision training (FP16)
- [ ] Implement beam search decoding
- [ ] Add fine-tuning capabilities
- [ ] Support for custom datasets
- [ ] Multi-GPU training support
- [ ] Model quantization for deployment

## License

MIT License

## Acknowledgments

- Hugging Face Transformers
- WikiText-103 Dataset
- Google Colab for free GPU access
