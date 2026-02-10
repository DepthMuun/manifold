import torch
import pytest
import numpy as np
from gfn.model import Manifold
from gfn.cuda import ops

def get_numerical_grad(model, inputs, param_name, eps=1e-3):
    """
    Computes numerical gradient of total loss w.r.t a parameter.
    """
    param = dict(model.named_parameters())[param_name]
    orig_data = param.data.clone()
    
    grad_num = torch.zeros_like(param)
    
    # Flatten for iteration
    flat_grad = grad_num.view(-1)
    
    for i in range(orig_data.numel()):
        with torch.no_grad():
            param.data.view(-1)[i] = orig_data.view(-1)[i] + eps
        logits_p = model(inputs)[0]
        loss_p = logits_p.pow(2).sum()
        
        with torch.no_grad():
            param.data.view(-1)[i] = orig_data.view(-1)[i] - eps
        logits_m = model(inputs)[0]
        loss_m = logits_m.pow(2).sum()
        
        flat_grad[i] = (loss_p - loss_m) / (2 * eps)
        with torch.no_grad():
            param.data.view(-1)[i] = orig_data.view(-1)[i]
        
    return grad_num

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("topology", ["low_rank", "torus"])
def test_cuda_adjoint_consistency(topology, metrics):
    """
    Compares analytical gradients from CUDA kernel with numerical gradients.
    """
    if not ops.CUDA_AVAILABLE:
        pytest.skip("CUDA kernels not available")

    dim = 16
    rank = 4
    batch = 2
    seq_len = 3
    
    physics_config = {
        "topology": {"type": "torus" if topology == "torus" else "euclidean"}
    }
    model = Manifold(
        vocab_size=10,
        dim=dim,
        depth=1,
        heads=1,
        rank=rank,
        integrator_type='leapfrog',
        physics_config=physics_config,
        use_scan=False
    ).cuda()
    
    # Enable gradients for critical parameters
    for p in model.parameters():
        p.requires_grad = True
        
    inputs = torch.randint(0, 10, (batch, seq_len)).cuda()
    
    # 1. Analytical Gradient (CUDA)
    logits = model(inputs)[0]
    loss = logits.pow(2).sum()
    loss.backward()
    
    results = {}
    
    def resolve_param(name_candidates):
        param_map = dict(model.named_parameters())
        for name in name_candidates:
            if name in param_map:
                return name
        return None

    param_targets = []
    if topology == "low_rank":
        u_name = resolve_param(["layers.0.christoffels.0.U", "layers.0.christoffel_adapter.U"])
        w_name = resolve_param(["layers.0.christoffels.0.W", "layers.0.christoffel_adapter.W"])
        if u_name is not None:
            param_targets.append(u_name)
        if w_name is not None:
            param_targets.append(w_name)

    forget_name = resolve_param(["layers.0.christoffels.0.forget_gate.weight", "layers.0.forget_gate.weight"])
    if forget_name is not None:
        param_targets.append(forget_name)
    
    # Check singularity parameters if active
    if hasattr(model.layers[0], 'singularity'):
        sing_name = resolve_param(["layers.0.singularity.weight"])
        if sing_name is not None:
            param_targets.append(sing_name)

    for p_name in param_targets:
        analytical = dict(model.named_parameters())[p_name].grad
        if analytical is None: continue
        
        numerical = get_numerical_grad(model, inputs, p_name)
        
        cos_sim = torch.nn.functional.cosine_similarity(analytical.flatten(), numerical.flatten(), dim=0)
        rel_err = torch.norm(analytical - numerical) / (torch.norm(numerical) + 1e-9)
        
        results[f"{p_name}_cos_sim"] = cos_sim.item()
        results[f"{p_name}_rel_err"] = rel_err.item()
        
        metrics.log(f"{p_name}_cos_sim", cos_sim)
        metrics.log(f"{p_name}_rel_err", rel_err)
        
        print(f"\n[{p_name}] Cosine Similarity: {cos_sim:.6f}, Rel Error: {rel_err:.6e}")
        
        # Threshold: Adjoint should be very accurate for short sequences
        # Leapfrog with high plasticity might have some drift, but > 0.99 cos_sim is expected.
        assert cos_sim > 0.95, f"Gradient mismatch in {p_name}: CosSim={cos_sim}"

    metrics.log("topology", topology)
    metrics.finish()
