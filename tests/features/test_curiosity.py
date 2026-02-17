import torch
import torch.nn as nn
from gfn.noise.curiosity import CuriosityNoise
from gfn.layers.base import MLayer

def test_curiosity_noise_flow():
    dim = 64
    heads = 4
    
    # Enable curiosity noise in physics_config
    physics_config = {
        'active_inference': {
            'curiosity_noise': {
                'enabled': True,
                'base_std': 0.1,
                'sensitivity': 1.0
            }
        }
    }
    
    layer = MLayer(dim, heads=heads, physics_config=physics_config)
    
    x = torch.randn(8, dim)
    v = torch.randn(8, dim)
    force = torch.randn(8, dim)
    
    # 1. Check that noise is injected during training
    layer.train()
    x_next_tr, v_next_tr, _, _ = layer(x, v, force=force)
    
    # 2. Check that noise is NOT injected during eval (by default)
    layer.eval()
    x_next_ev, v_next_ev, _, _ = layer(x, v, force=force)
    
    print(f"Train V std: {v_next_tr.std().item():.4f}")
    print(f"Eval V std: {v_next_ev.std().item():.4f}")
    
    # 3. Check scaling with force
    # High force should lead to higher variance in velocity delta
    layer.train()
    v_base = torch.zeros(100, dim)
    x_base = torch.zeros(100, dim)
    
    # Low force
    force_low = torch.zeros(100, dim)
    _, v_low, _, _ = layer(x_base, v_base, force=force_low)
    
    # High force
    force_high = torch.ones(100, dim) * 10.0
    _, v_high, _, _ = layer(x_base, v_base, force=force_high)
    
    std_low = v_low.std().item()
    std_high = v_high.std().item()
    
    print(f"Low Force Std: {std_low:.4f}")
    print(f"High Force Std: {std_high:.4f}")
    
    assert std_high > std_low, "High force should trigger more curiosity noise!"
    print("✓ Curiosity Noise scaling verified.")

if __name__ == "__main__":
    test_curiosity_noise_flow()
