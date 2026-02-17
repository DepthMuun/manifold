import os
import sys
import torch

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root not in sys.path:
    sys.path.insert(0, root)

from gfn.cuda.ops import christoffel_fused, ChristoffelOperation, CUDA_AVAILABLE
from gfn.cuda.core import CudaConstants


def manual_christoffel(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology):
    h = torch.matmul(v, U)
    energy = torch.sum(h * h, dim=-1, keepdim=True) / max(1, h.shape[-1])
    scale = 1.0 / (1.0 + torch.sqrt(energy) + CudaConstants.EPSILON_STANDARD)
    M = 1.0
    if plasticity != 0.0:
        v_energy = torch.sum(v * v, dim=-1, keepdim=True) / max(1, v.shape[-1])
        M = 1.0 + plasticity * 0.1 * torch.tanh(v_energy)
    if x is not None and V_w is not None and V_w.numel() > 0:
        if topology == 1:
            pot = torch.sum(torch.sin(x) * V_w, dim=-1, keepdim=True)
        else:
            pot = torch.sum(x * V_w, dim=-1, keepdim=True)
        gate = torch.sigmoid(pot)
        soft_m = torch.sigmoid(CudaConstants.SINGULARITY_GATE_SLOPE * (gate - sing_thresh))
        M = M * (1.0 + (sing_strength - 1.0) * soft_m)
    gamma = torch.matmul(h * h, W.t()) * scale * M
    gamma = CudaConstants.CURVATURE_CLAMP * torch.tanh(gamma / CudaConstants.CURVATURE_CLAMP)
    return gamma


def compute_grads(res, v, U, W):
    for t in (v, U, W):
        if t.grad is not None:
            t.grad.zero_()
    loss = res.pow(2).sum()
    loss.backward()
    return v.grad.clone(), U.grad.clone(), W.grad.clone()


def run_case(name, v_base, U_base, W_base, x_base, Vw_full, Vw_empty, plasticity, sing_thresh, sing_strength, topology):
    v = v_base.detach().clone().requires_grad_(True)
    U = U_base.detach().clone().requires_grad_(True)
    W = W_base.detach().clone().requires_grad_(True)
    x = x_base.detach().clone().requires_grad_(True)

    V_w = Vw_full if (Vw_full is not None and Vw_full.numel() > 0) else Vw_empty
    christoffel_op = ChristoffelOperation({
        "curvature_clamp": CudaConstants.CURVATURE_CLAMP,
        "epsilon": CudaConstants.EPSILON_STANDARD,
        "singularity_gate_slope": CudaConstants.SINGULARITY_GATE_SLOPE
    })

    res_manual = manual_christoffel(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology)
    res_op = christoffel_op.forward(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology)
    res_cuda = christoffel_fused(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology)

    fwd_manual_op = (res_manual - res_op).abs().max().item()
    fwd_manual_cuda = (res_manual - res_cuda).abs().max().item()
    fwd_op_cuda = (res_op - res_cuda).abs().max().item()

    grad_v_m, grad_U_m, grad_W_m = compute_grads(res_manual, v, U, W)
    grad_v_o, grad_U_o, grad_W_o = compute_grads(res_op, v, U, W)
    grad_v_c, grad_U_c, grad_W_c = compute_grads(res_cuda, v, U, W)

    gv_m_c = (grad_v_m - grad_v_c).abs().max().item()
    gU_m_c = (grad_U_m - grad_U_c).abs().max().item()
    gW_m_c = (grad_W_m - grad_W_c).abs().max().item()

    gv_o_c = (grad_v_o - grad_v_c).abs().max().item()
    gU_o_c = (grad_U_o - grad_U_c).abs().max().item()
    gW_o_c = (grad_W_o - grad_W_c).abs().max().item()

    print(f"case={name}")
    print(f"  fwd manual-op:   {fwd_manual_op:.6f}")
    print(f"  fwd manual-cuda: {fwd_manual_cuda:.6f}")
    print(f"  fwd op-cuda:     {fwd_op_cuda:.6f}")
    print(f"  grad v m-c:      {gv_m_c:.6f}")
    print(f"  grad U m-c:      {gU_m_c:.6f}")
    print(f"  grad W m-c:      {gW_m_c:.6f}")
    print(f"  grad v o-c:      {gv_o_c:.6f}")
    print(f"  grad U o-c:      {gU_o_c:.6f}")
    print(f"  grad W o-c:      {gW_o_c:.6f}")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def run_suite(batch, dim, rank):
    device = torch.device("cuda")
    v_base = torch.randn(batch, dim, device=device)
    U_base = torch.randn(dim, rank, device=device)
    W_base = torch.randn(dim, rank, device=device)
    x_base = torch.randn(batch, dim, device=device)
    Vw_full = torch.randn(1, dim, device=device)
    Vw_empty = torch.empty(0, device=device)

    header = f"batch={batch} dim={dim} rank={rank}"
    print("")
    print(header)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    run_case("base", v_base, U_base, W_base, x_base, Vw_empty, Vw_empty, 0.0, 0.2, 1.0, 0)
    run_case("plasticity", v_base, U_base, W_base, x_base, Vw_empty, Vw_empty, 0.5, 0.2, 1.0, 0)
    run_case("singularity", v_base, U_base, W_base, x_base, Vw_full, Vw_empty, 0.0, 0.2, 5.0, 0)
    run_case("both", v_base, U_base, W_base, x_base, Vw_full, Vw_empty, 0.5, 0.2, 5.0, 0)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def main():
    if not CUDA_AVAILABLE:
        print("CUDA extension not loaded")
        return

    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    run_suite(2, 32, 8)
    use_large = os.getenv("CHRISTOFFEL_LARGE", "0") == "1"
    if use_large:
        run_suite(4, 64, 16)
        run_suite(16, 128, 32)


if __name__ == "__main__":
    main()
