"""
Checkpoint to JSON Converter
Converts PyTorch .pt checkpoint files to JSON format for Kaggle upload
"""

import torch
import json
import numpy as np
import argparse
from pathlib import Path


def tensor_to_list(tensor):
    """Convert tensor to nested list"""
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy().tolist()
    return tensor


def convert_state_dict_to_json(state_dict):
    """Convert model state dict to JSON-serializable format"""
    json_dict = {}
    
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            json_dict[key] = {
                'data': tensor_to_list(value),
                'shape': list(value.shape),
                'dtype': str(value.dtype)
            }
        else:
            json_dict[key] = value
    
    return json_dict


def convert_checkpoint_to_json(checkpoint_path, output_path=None):
    """
    Convert PyTorch checkpoint to JSON format
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        output_path: Path to save JSON file (optional)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Convert to JSON-serializable format
    json_checkpoint = {}
    
    # Handle model state dict
    if 'model_state_dict' in checkpoint:
        print("Converting model state dict...")
        json_checkpoint['model_state_dict'] = convert_state_dict_to_json(
            checkpoint['model_state_dict']
        )
    
    # Handle optimizer state dict
    if 'optimizer_state_dict' in checkpoint:
        print("Converting optimizer state dict...")
        optimizer_state = checkpoint['optimizer_state_dict']
        json_checkpoint['optimizer_state_dict'] = {
            'state': {},
            'param_groups': optimizer_state.get('param_groups', [])
        }
        
        # Convert optimizer state tensors
        for param_id, param_state in optimizer_state.get('state', {}).items():
            json_checkpoint['optimizer_state_dict']['state'][str(param_id)] = {}
            for key, value in param_state.items():
                if isinstance(value, torch.Tensor):
                    json_checkpoint['optimizer_state_dict']['state'][str(param_id)][key] = {
                        'data': tensor_to_list(value),
                        'shape': list(value.shape),
                        'dtype': str(value.dtype)
                    }
                else:
                    json_checkpoint['optimizer_state_dict']['state'][str(param_id)][key] = value
    
    # Handle other metadata
    for key in ['epoch', 'step', 'val_loss', 'val_perplexity', 'train_loss']:
        if key in checkpoint:
            json_checkpoint[key] = checkpoint[key]
    
    # Determine output path
    if output_path is None:
        output_path = Path(checkpoint_path).stem + '.json'
    
    print(f"Saving JSON to: {output_path}")
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(json_checkpoint, f, indent=2)
    
    # Get file sizes
    checkpoint_size = Path(checkpoint_path).stat().st_size / (1024 * 1024)
    json_size = Path(output_path).stat().st_size / (1024 * 1024)
    
    print(f"\n✓ Conversion complete!")
    print(f"Original checkpoint size: {checkpoint_size:.2f} MB")
    print(f"JSON file size: {json_size:.2f} MB")
    print(f"Size ratio: {json_size/checkpoint_size:.2f}x")
    
    return output_path


def load_json_checkpoint(json_path):
    """
    Load checkpoint from JSON format back to PyTorch
    
    Args:
        json_path: Path to JSON checkpoint file
    
    Returns:
        checkpoint: PyTorch checkpoint dictionary
    """
    print(f"Loading JSON checkpoint from: {json_path}")
    
    with open(json_path, 'r') as f:
        json_checkpoint = json.load(f)
    
    checkpoint = {}
    
    # Convert model state dict back to tensors
    if 'model_state_dict' in json_checkpoint:
        print("Converting model state dict to tensors...")
        checkpoint['model_state_dict'] = {}
        for key, value in json_checkpoint['model_state_dict'].items():
            if isinstance(value, dict) and 'data' in value:
                tensor_data = torch.tensor(value['data'])
                checkpoint['model_state_dict'][key] = tensor_data
            else:
                checkpoint['model_state_dict'][key] = value
    
    # Convert optimizer state dict back to tensors
    if 'optimizer_state_dict' in json_checkpoint:
        print("Converting optimizer state dict to tensors...")
        checkpoint['optimizer_state_dict'] = {
            'state': {},
            'param_groups': json_checkpoint['optimizer_state_dict'].get('param_groups', [])
        }
        
        for param_id, param_state in json_checkpoint['optimizer_state_dict'].get('state', {}).items():
            checkpoint['optimizer_state_dict']['state'][int(param_id)] = {}
            for key, value in param_state.items():
                if isinstance(value, dict) and 'data' in value:
                    tensor_data = torch.tensor(value['data'])
                    checkpoint['optimizer_state_dict']['state'][int(param_id)][key] = tensor_data
                else:
                    checkpoint['optimizer_state_dict']['state'][int(param_id)][key] = value
    
    # Copy metadata
    for key in ['epoch', 'step', 'val_loss', 'val_perplexity', 'train_loss']:
        if key in json_checkpoint:
            checkpoint[key] = json_checkpoint[key]
    
    print("✓ JSON checkpoint loaded successfully!")
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PyTorch checkpoint to JSON')
    parser.add_argument('checkpoint', type=str, help='Path to .pt checkpoint file')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file path')
    parser.add_argument('--test-load', action='store_true', help='Test loading the JSON back')
    
    args = parser.parse_args()
    
    # Convert to JSON
    json_path = convert_checkpoint_to_json(args.checkpoint, args.output)
    
    # Test loading if requested
    if args.test_load:
        print("\n" + "="*50)
        print("Testing JSON load...")
        print("="*50)
        loaded_checkpoint = load_json_checkpoint(json_path)
        print(f"Loaded checkpoint keys: {list(loaded_checkpoint.keys())}")
        if 'model_state_dict' in loaded_checkpoint:
            print(f"Model state dict keys: {len(loaded_checkpoint['model_state_dict'])}")
