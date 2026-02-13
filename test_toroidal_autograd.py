#!/usr/bin/env python3
"""
Test toroidal autograd implementation for numerical consistency.
"""

import torch
import torch.nn as nn
import math
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gfn.cuda.autograd import recurrent_manifold_fused_autograd

def test_toroidal_autograd():
    print("=== TESTING TOROIDAL AUTOGRAD IMPLEMENTATION ===")
    
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available, testing on CPU")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    
    # Test parameters
    B, T, D, H = 2, 50, 8, 2  # batch, time, dim, heads
    num_layers = 2
    dt = 0.1
    
    # Create inputs that will drift beyond 2π
    x = torch.zeros(B, D, device=device, requires_grad=True)
    v = torch.ones(B, D, device=device, requires_grad=True) * 10.0  # High velocity
    f = torch.zeros(B, T, D, device=device, requires_grad=True)
    
    # Create dummy U, W matrices (low-rank decomposition)
    rank = 4
    U_stack = torch.randn(num_layers * H * D, rank, device=device) * 0.01
    W_stack = torch.randn(num_layers * H * rank, D, device=device) * 0.01
    
    # dt scales and forget rates - match num_layers
    dt_scales = torch.ones(num_layers, device=device)
    forget_rates = torch.zeros(num_layers, device=device)
    
    # Optional parameters (set to None)
    mix_x = None
    mix_v = None
    Wf = None
    Wi = None
    bf = None
    Wp = None
    bp = None
    
    print(f"Initial x range: [{x.min().item():.3f}, {x.max().item():.3f}]")
    print(f"Initial v range: [{v.min().item():.3f}, {v.max().item():.3f}]")
    print(f"num_layers: {num_layers}, dt_scales shape: {dt_scales.shape}")
    
    # Test 1: Euclidean topology (should drift beyond 2π)
    print("\n--- Testing Euclidean Topology ---")
    x_euc, v_euc, x_seq_euc, _, _ = recurrent_manifold_fused_autograd(
        x=x, v=v, f=f,
        U_stack=U_stack, W_stack=W_stack,
        dt=dt, dt_scales=dt_scales, forget_rates=forget_rates,
        num_heads=H, topology=0,  # Euclidean
        plasticity=0.0, sing_thresh=0.5, sing_strength=2.0,
        mix_x=mix_x, mix_v=mix_v, Wf=Wf, Wi=Wi, bf=bf, Wp=Wp, bp=bp
    )
    
    print(f"Final x range: [{x_seq_euc.min().item():.3f}, {x_seq_euc.max().item():.3f}]")
    max_euc = x_seq_euc.max().item()
    if max_euc > 2 * math.pi:
        print(f"✓ Euclidean drifted beyond 2π (max: {max_euc:.3f})")
    else:
        print(f"⚠ Euclidean didn't drift enough (max: {max_euc:.3f})")
    
    # Test 2: Toroidal topology (should wrap to [0, 2π))
    print("\n--- Testing Toroidal Topology ---")
    x_tor, v_tor, x_seq_tor, _, _ = recurrent_manifold_fused_autograd(
        x=x, v=v, f=f,
        U_stack=U_stack, W_stack=W_stack,
        dt=dt, dt_scales=dt_scales, forget_rates=forget_rates,
        num_heads=H, topology=1,  # Torus
        plasticity=0.0, sing_thresh=0.5, sing_strength=2.0,
        mix_x=mix_x, mix_v=mix_v, Wf=Wf, Wi=Wi, bf=bf, Wp=Wp, bp=bp
    )
    
    print(f"Final x range: [{x_seq_tor.min().item():.3f}, {x_seq_tor.max().item():.3f}]")
    max_tor = x_seq_tor.max().item()
    min_tor = x_seq_tor.min().item()
    
    # Check wrapping
    TWO_PI = 2 * math.pi
    if max_tor <= TWO_PI + 0.01 and min_tor >= -0.01:
        print(f"✓ Toroidal wrapping successful (bounded to [0, 2π])")
        print(f"  Range: [{min_tor:.3f}, {max_tor:.3f}], 2π = {TWO_PI:.3f}")
    else:
        print(f"✗ Toroidal wrapping failed (range: [{min_tor:.3f}, {max_tor:.3f}])")
        print(f"  Expected: [0, {TWO_PI:.3f}]")
    
    # Test 3: Compare trajectories (should be different)
    print("\n--- Comparing Trajectories ---")
    diff = torch.abs(x_seq_euc - x_seq_tor).mean().item()
    print(f"Mean absolute difference: {diff:.6f}")
    
    if diff > 0.1:
        print("✓ Trajectories are significantly different")
    else:
        print("⚠ Trajectories are very similar (wrapping may not be working)")
    
    # Test 4: Check wrapping function directly
    print("\n--- Testing Wrapping Function ---")
    test_values = torch.tensor([0.0, math.pi, 2*math.pi, 3*math.pi, -math.pi, -2*math.pi], device=device)
    wrapped = torch.fmod(test_values, 2 * torch.pi)
    wrapped = torch.where(wrapped < 0, wrapped + 2 * torch.pi, wrapped)
    print("Test values:", test_values.cpu().numpy())
    print("Wrapped values:", wrapped.cpu().numpy())
    print("Expected in [0, 2π]:", (wrapped >= 0).all().item() and (wrapped <= 2*math.pi).all().item())
    
    # Test 5: Numerical stability
    print("\n--- Testing Numerical Stability ---")
    
    # Check for NaN or Inf
    if torch.isnan(x_seq_tor).any() or torch.isinf(x_seq_tor).any():
        print("✗ Numerical instability detected (NaN/Inf)")
    else:
        print("✓ No numerical instability detected")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_toroidal_autograd()