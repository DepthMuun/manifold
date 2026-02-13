
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from gfn.geometry.gauge import GaugeChristoffel, gauge_invariant_loss

def test_gauge_forward():
    print("Testing GaugeChristoffel Forward Pass...")
    dim = 16
    gauge_dim = 4
    batch_size = 5
    
    model = GaugeChristoffel(dim, gauge_dim, group='U1')
    
    v = torch.randn(batch_size, dim)
    x = torch.randn(batch_size, dim)
    
    # 1. Test Connection Computation
    A = model.compute_connection(x)
    assert A.shape == (batch_size, dim, gauge_dim)
    print(f"Connection shape: {A.shape} [OK]")
    
    # 2. Test Parallel Transport
    v_trans = model.parallel_transport(v, x)
    assert v_trans.shape == v.shape
    print(f"Transported v shape: {v_trans.shape} [OK]")
    
    # 3. Test Full Forward (Christoffel)
    gamma = model(v, x)
    assert gamma.shape == (batch_size, dim, dim)
    print(f"Gamma shape: {gamma.shape} [OK]")
    
    # 4. Test Field Strength (expensive, so small batch)
    F = model.compute_field_strength(x[:2])
    # Expected shape: [batch, dim, dim, gauge_dim] for Abelian
    # Actually my implementation returns [batch, dim, dim, gauge_dim] stacked? 
    # Let's check the code: it appends to list then stacks.
    # Ah, the logic inside was: d_A [dim, dim, gauge_dim].
    # So `F` should be [batch, dim, dim, gauge_dim]
    print(f"Field strength shape: {F.shape}")
    assert F.shape == (2, dim, dim, gauge_dim)
    print(f"Field Strength shape: {F.shape} [OK]")
    
    print("GaugeChristoffel Test PASSED.")

if __name__ == "__main__":
    test_gauge_forward()
