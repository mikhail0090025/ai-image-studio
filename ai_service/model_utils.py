import torch
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic
import inspect
import os
import hashlib

def quantize_model(model, name):
    print(f"Quantizing {name}...")

    try:
        model = quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )

        print(f"✓ {name} quantized")
        return model

    except Exception as e:
        print(f"⚠ {name}: {e}")
        return model

def hash_model(model):
    h = hashlib.sha256()

    for k, v in model.state_dict().items():
        if isinstance(v, torch.Tensor):
            h.update(v.cpu().numpy().tobytes())

    return h.hexdigest()