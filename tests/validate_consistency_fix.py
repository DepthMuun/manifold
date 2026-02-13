
import torch
import sys
import os
from pathlib import Path

# Add project root
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from gfn.core.manifold import Manifold
from gfn.core.adjoint import AdjointManifold
from gfn.cuda import ops

def test_ops_repair():
    print("\n--- Testing gfn.cuda.ops repair ---")
    
    # 1. Check if heun_fused is available in ops
    if hasattr(ops, 'heun_fused'):
        print("✓ heun_fused is now available in ops")
    else:
        print("✗ heun_fused MISSING in ops")
        return False
        
    # 2. Check velocity_friction_scale in signature
    import inspect
    sig = inspect.signature(ops.heun_fused)
    if 'velocity_friction_scale' in sig.parameters:
        print("✓ heun_fused accepts velocity_friction_scale")
    else:
        print("✗ heun_fused MISSING velocity_friction_scale parameter")
        return False
        
    # 3. Test execution (Fallback mode)
    B, D = 2, 16
    x = torch.randn(B, D)
    v = torch.randn(B, D)
    f = torch.randn(B, D) # Force for one step (simplified)
    U = torch.randn(D, 4)
    W = torch.randn(D, 4)
    
    try:
        x_out, v_out = ops.heun_fused(
            x, v, f, U, W, 
            dt=0.1, dt_scale=1.0, steps=1, 
            topology=0, 
            velocity_friction_scale=0.1
        )
        print("✓ heun_fused execution successful")
    except Exception as e:
        print(f"✗ heun_fused execution failed: {e}")
        return False
        
    return True

def test_adjoint_fix():
    print("\n--- Testing AdjointManifold fix ---")
    
    vocab_size = 100
    dim = 32
    
    model = AdjointManifold(vocab_size, dim, depth=1)
    
    input_ids = torch.randint(0, vocab_size, (2, 5))
    
    try:
        out = model(input_ids)
        
        # Check return signature
        if len(out) == 6:
            print("✓ AdjointManifold returns 6-element tuple (API parity)")
            logits, state, _, v_seq, x_seq, forces = out
            print(f"  - Logits shape: {logits.shape}")
            print(f"  - x_seq shape: {x_seq.shape}")
            print(f"  - v_seq shape: {v_seq.shape}")
        else:
            print(f"✗ AdjointManifold returned {len(out)} elements (Expected 6)")
            return False
            
    except Exception as e:
        print(f"✗ AdjointManifold forward failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    print("Running Consistency Verification...")
    
    success_ops = test_ops_repair()
    success_adj = test_adjoint_fix()
    
    if success_ops and success_adj:
        print("\n[SUCCESS] All critical consistency fixes verified.")
        sys.exit(0)
    else:
        print("\n[FAILURE] Some checks failed.")
        sys.exit(1)
