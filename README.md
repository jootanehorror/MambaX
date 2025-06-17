# MambaX

PyTorch implementation of the Mamba architecture with enhanced production-ready features:

1. **ONNX Export** - Full model export support for deployment
2. **Chunk Processing** - Single-forwardpass chunk handling (no token loops)
3. **CPU-First** - Optimized execution without CUDA dependencies

## Key Advantages

- **Production Ready**: ONNX-compatible for serving
- **No Token Loops**: Processes entire chunks in single forward pass
- **Hardware Agnostic**: Runs equally well on CPU/GPU

## Acknowledgements
Builds upon reference work from [alxndrTL/mamba.py](https://github.com/alxndrTL/mamba.py)

## Usage

### 1. Standard Forward Pass
```python
import torch
import torch.nn as nn
from mamba import Mamba

# Initialize model
model = Mamba(
    d_model=512,
    d_inner=1024,
    d_conv=4,
    d_state=16,
    dt_rank=64,
    act=nn.SiLU()
)

# Process full sequence
x = torch.rand(1, 13, 512)  # (batch, seq_len, dim)
output = model(x)  # single forward pass
```

### 2. Single Token Processing
```python

x_token = torch.rand(1, 1, 512)  # (batch, 1, dim)
output, new_state, new_conv = model(x_token, state_cache, conv_cache)

```

### 3. Chunk-Based Processing
```python

# Initialize with empty caches
state_cache = torch.zeros(1, 1024, 16)  # (batch, d_inner, d_state)
conv_cache = torch.zeros(1, 1024, 3)     # (batch, d_inner, d_conv-1)

# Process chunks (e.g. 8 tokens at once)
x_chunk = torch.rand(1, 8, 512)
output, new_state, new_conv = model(x_chunk, state_cache, conv_cache)
```
