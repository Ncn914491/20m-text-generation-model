import traceback
try:
    import torch
    import json
    import numpy as np
    print("Imports successful")
    checkpoint = torch.load("checkpoint_epoch1_step6500.pt", map_location='cpu')
    print(f"Checkpoint loaded, keys: {list(checkpoint.keys())}")
except Exception as e:
    traceback.print_exc()
    print(f"Error: {e}")
