#!/usr/bin/env python3
"""
Debug the toroidal autograd implementation.
"""

import torch
import math
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gfn.cuda.autograd import recurrent_manifold_fused_autograd

def debug_autograd():
    print("=== DEBUGGING TOROIDAL AUTOGRAD ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simple parameters
    B, T, D, H = 1, 10, 4, 2  # batch, time, dim, heads
    num_layers = 2
    dt = 0.1
    
    print(f"B={B}, T={T}, D={D}, H={H}, num_layers={num_layers}")
    
    # Create simple inputs
    x = torch.zeros(B, D, device=device, requires_grad=True)
    v = torch.ones(B, D, device=device, requires_grad=True) * 5.0
    f = torch.zeros(B, T, D, device=device, requires_grad=True)
    
    # Simple U, W matrices
    rank = 2
    U_stack = torch.ones(num_layers * H * D, rank, device=device) * 0.001
    W_stack = torch.ones(num_layers * H * rank, D, device=device) * 0.001
    
    # dt scales - this is the key issue
    dt_scales = torch.ones(num_layers, device=device)  # Shape: [num_layers]
    forget_rates = torch.zeros(num_layers, device=device)
    
    print(f"dt_scales shape: {dt_scales.shape}")
    print(f"dt_scales values: {dt_scales}")
    
    # Test each layer index manually
    for layer_idx in range(num_layers):
        print(f"layer_idx={layer_idx}, dt_scales[{layer_idx}] = {dt_scales[layer_idx]}")
    
    # Try a minimal call
    try:
        result = recurrent_manifold_fused_autograd(
            x=x, v=v, f=f,
            U_stack=U_stack, W_stack=W_stack,
            dt=dt, dt_scales=dt_scales, forget_rates=forget_rates,
            num_heads=H, topology=0,
            plasticity=0.0, sing_thresh=0.5, sing_strength=2.0,
            mix_x=None, mix_v=None, Wf=None, Wi=None, bf=None, Wp=None, bp=None
        )
        print("✓ Success!")
        print(f"Result shapes: x={result[0].shape}, v={result[1].shape}, seq={result[2].shape}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_autograd()