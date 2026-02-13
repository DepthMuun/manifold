import torch
import torch.nn as nn
from gfn.core.manifold import Manifold

def test_hysteresis_geometric_flow():
    vocab_size = 100
    dim = 64
    heads = 4
    
    physics_config = {
        'hysteresis': {
            'enabled': True,
            'rank': 8
        }
    }
    
    model = Manifold(vocab_size, dim=dim, heads=heads, physics_config=physics_config)
    model.train()
    
    input_ids = torch.randint(0, vocab_size, (1, 10))
    
    # 1. Run forward once to accumulate memory
    logits1, state1, _, _, _, _ = model(input_ids)
    h_state1 = model._hysteresis_state.clone()
    
    # 2. Run forward again with DIFFERENT input but same initial state
    # The presence of h_state1 should make it different from a fresh run
    input_ids2 = torch.randint(0, vocab_size, (1, 10))
    
    # Fresh model for baseline
    model_fresh = Manifold(vocab_size, dim=dim, heads=heads, physics_config=physics_config)
    model_fresh.train()
    
    logits_fresh, _, _, _, _, _ = model_fresh(input_ids2)
    
    # Model with memory
    logits_mem, _, _, _, _, _ = model(input_ids2)
    
    # Hysteresis state should have updated
    h_state2 = model._hysteresis_state
    
    print(f"Hysteresis state 1 norm: {h_state1.norm().item():.4f}")
    print(f"Hysteresis state 2 norm: {h_state2.norm().item():.4f}")
    
    # The logits should differ from fresh because of memory (though weights are random and small)
    # But h_state should definitely be non-zero
    assert h_state1.norm() > 0, "Memory state should be non-zero after forward pass"
    assert not torch.allclose(h_state1, h_state2), "Memory state should evolve"
    
    print("✓ Geometric Hysteresis evolution verified.")

if __name__ == "__main__":
    test_hysteresis_geometric_flow()
