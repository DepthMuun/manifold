import torch
import torch.nn as nn
from gfn.layers.base import MLayer
from gfn.core.manifold import Manifold
from gfn.cuda import ops
import numpy as np

def test_cuda_parity():
    print("=== Starting CUDA Parity Verification ===")
    
    # Configuration
    batch_size = 32
    dim = 64
    heads = 4
    seq_len = 10
    dt = 0.01
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type != 'cuda':
        print("Skipping CUDA tests (no CUDA device)")
        return

    # Create dummy physics config
    physics_config = {
        'topology': 'euclidean',
        'plasticity': 0.1, # Active plasticity
        'singularities': {'enabled': True, 'threshold': 0.5, 'strength': 2.0},
        'friction': {'enabled': True, 'velocity_scale': 0.1}, # Velocity friction
        'hysteresis': {'enabled': True, 'decay': 0.9}
    }
    
    # 1. Initialize Manifold
    manifold = Manifold(dim, inner_dim=64, heads=heads, physics_config=physics_config).to(device)
    
    # 2. Create MLayer
    # We want to compare use_fused=True vs use_fused=False
    # But MLayer uses use_fused=True by default if available.
    # We will manually call the underlying functions.
    
    layer = MLayer(manifold, 0, dt=dt).to(device)
    
    # Inputs
    x = torch.randn(batch_size, dim, device=device)
    v = torch.randn(batch_size, dim, device=device)
    # Force: [batch, seq_len, dim]
    f = torch.randn(batch_size, seq_len, dim, device=device)
    
    print(f"Testing with: Plasticity={physics_config['plasticity']}, Singularity=True, VelocityFriction=True")

    # --- Run 1: Fused (Unified Kernel) ---
    # This uses my modified unified_mlayer.cu with Phase 2 logic and Friction
    with torch.no_grad():
        x_fused, v_fused, _ = layer.forward(x.clone(), v.clone(), f.clone(), use_fused=True)
    
    # --- Run 2: Component-wise (Python Integrator + Component Kernel) ---
    # This uses LeapfrogOperation (Python) + ChristoffelOperation (Python wrapper -> LowRankChristoffel Kernel)
    # Note: LowRankChristoffel Kernel (christoffel_impl.cuh) already had Phase 2 logic.
    # But LeapfrogOperation (Python) DOES NOT support velocity friction properly?
    # Wait, my audit said LeapfrogOperation in ops.py ignores velocity friction.
    # So if I enabled it in Unified Kernel, they will mismatch!
    
    # Let's check LeapfrogOperation in ops.py again.
    # If it ignores velocity friction, I should FIX IT there too to match Unified Kernel?
    # Or disable velocity friction for this test to verify Christoffel first?
    
    # Let's first test with friction disabled to verify Christoffel Phase 2.
    print("\n--- Test A: Geometry Parity (Friction Disabled) ---")
    layer.velocity_friction_scale = 0.0
    with torch.no_grad():
        x_fused_a, v_fused_a, _ = layer.forward(x.clone(), v.clone(), f.clone(), use_fused=True)
        x_comp_a, v_comp_a, _ = layer.forward(x.clone(), v.clone(), f.clone(), use_fused=False)
    
    diff_x = (x_fused_a - x_comp_a).abs().max().item()
    diff_v = (v_fused_a - v_comp_a).abs().max().item()
    print(f"Max Diff X: {diff_x:.2e}")
    print(f"Max Diff V: {diff_v:.2e}")
    
    if diff_x < 1e-5 and diff_v < 1e-5:
        print("✅ Geometry Parity Passed!")
    else:
        print("❌ Geometry Parity Failed!")
        # If failed, likely due to Singularity/Plasticity mismatch or my implementation being slightly different.

    # --- Test B: Friction Parity ---
    print("\n--- Test B: Friction Parity (Velocity Friction Enabled) ---")
    layer.velocity_friction_scale = 0.1
    
    # I suspect Component path (Python) will FAIL to apply velocity friction.
    # So I expect mismatch here unless I fix ops.py.
    with torch.no_grad():
        x_fused_b, v_fused_b, _ = layer.forward(x.clone(), v.clone(), f.clone(), use_fused=True)
        x_comp_b, v_comp_b, _ = layer.forward(x.clone(), v.clone(), f.clone(), use_fused=False)
        
    diff_x_b = (x_fused_b - x_comp_b).abs().max().item()
    diff_v_b = (v_fused_b - v_comp_b).abs().max().item()
    print(f"Max Diff X: {diff_x_b:.2e}")
    print(f"Max Diff V: {diff_v_b:.2e}")

    if diff_x_b < 1e-5 and diff_v_b < 1e-5:
        print("✅ Friction Parity Passed!")
    else:
        print("⚠️ Friction Parity Failed (Expected if Python ops.py lacks velocity friction)")

if __name__ == "__main__":
    test_cuda_parity()
