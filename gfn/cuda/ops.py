import torch
import torch.nn as nn
import os
import sys
import importlib.util
import importlib.machinery
from pathlib import Path

# CUDA op loading and Python fallbacks

CUDA_AVAILABLE = False
gfn_cuda = None

def get_cuda_path():
    return os.path.dirname(os.path.abspath(__file__))

# Specialized attempt to load/import the gfn_cuda module
_CUDA_LOG_ONCE = False

def _log_cuda_status():
    global _CUDA_LOG_ONCE
    if not _CUDA_LOG_ONCE:
        if CUDA_AVAILABLE:
            print(f"[GFN] CUDA enabled: {torch.cuda.get_device_name(0)}")
        else:
            print("[GFN] CUDA disabled: using Python fallbacks")
        _CUDA_LOG_ONCE = True

def _prepare_dll_paths():
    try:
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.exists():
            os.add_dll_directory(str(torch_lib))
    except Exception:
        pass
    for ver in ["v12.9", "v12.4", "v12.3", "v11.8"]:
        p = Path(f"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/{ver}/bin")
        if p.exists():
            try:
                os.add_dll_directory(str(p))
            except Exception:
                pass

cuda_dir = Path(__file__).resolve().parent
project_root = cuda_dir.parent.parent
if str(cuda_dir) not in sys.path:
    sys.path.insert(0, str(cuda_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def _load_local_gfn_cuda():
    global gfn_cuda, CUDA_AVAILABLE
    _prepare_dll_paths()
    candidates = list(cuda_dir.glob("gfn_cuda*.pyd")) + list(project_root.glob("gfn_cuda*.pyd"))
    for path in candidates:
        try:
            loader = importlib.machinery.ExtensionFileLoader("gfn_cuda", str(path))
            spec = importlib.util.spec_from_file_location("gfn_cuda", str(path), loader=loader)
            if spec is None:
                continue
            module = importlib.util.module_from_spec(spec)
            loader.exec_module(module)
            sys.modules["gfn_cuda"] = module
            gfn_cuda = module
            CUDA_AVAILABLE = True
            return True
        except Exception:
            continue
    return False

try:
    _prepare_dll_paths()
    import gfn_cuda
    CUDA_AVAILABLE = True
except ImportError:
    try:
        _prepare_dll_paths()
        from . import gfn_cuda
        CUDA_AVAILABLE = True
    except ImportError:
        CUDA_AVAILABLE = _load_local_gfn_cuda()

# Initialize status on first import if possible, or wait for ops
if CUDA_AVAILABLE:
    _log_cuda_status()

def christoffel_fused(v, U, W, x=None, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0, topology=0, R=2.0, r=1.0):
    """
    Christoffel projection with optional plasticity and periodic features.
    """
    if CUDA_AVAILABLE and v.is_cuda:
        from .autograd import christoffel_fused_autograd
        return christoffel_fused_autograd(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology, R, r)
    
    # Python fallback (vectorized)
    # print("[GFN:WARN] Fallback to Python implementation for Christoffel Fused")
    # h = U^T v
    h = torch.matmul(v, U) # [B, R]
    energy = torch.sum(h*h, dim=-1, keepdim=True)
    
    # Singular value gating
    if sing_strength > 0.0:
        gate = torch.sigmoid((energy - sing_thresh) * sing_strength)
        h = h * gate
        
    # Plasticity update (optional)
    if plasticity > 0.0 and x is not None and V_w is not None:
         # Simplified Hebbian-like update simulation for fallback
         pass

    # dv = U h
    dv = torch.matmul(h, W) # [B, D]
    return dv

# --- Head Mixing Autograd Function ---

class HeadMixingFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_heads, v_heads, W_x, W_v, topology):
        # Ensure contiguous
        x_heads = x_heads.contiguous()
        v_heads = v_heads.contiguous()
        
        ctx.save_for_backward(x_heads, v_heads, W_x, W_v)
        ctx.topology = topology
        
        if CUDA_AVAILABLE and x_heads.is_cuda:
            # Call the raw CUDA wrapper exposed in gfn_cuda
            # Returns [x_out, v_out] list/vector
            outs = gfn_cuda.head_mixing_fused(x_heads, v_heads, W_x, W_v, topology)
            return outs[0], outs[1]
        else:
             raise NotImplementedError("CPU fallback not implemented in ops.py for head_mixing_fused")

    @staticmethod
    def backward(ctx, grad_x_out, grad_v_out):
        x_heads, v_heads, W_x, W_v = ctx.saved_tensors
        topology = ctx.topology
        
        if CUDA_AVAILABLE and x_heads.is_cuda:
            heads = x_heads.size(0)
            grads = gfn_cuda.head_mixing_backward(
                grad_x_out.contiguous(), 
                grad_v_out.contiguous(), 
                x_heads, v_heads, W_x, W_v, 
                heads, topology
            )
            return grads[0], grads[1], grads[2], grads[3], None

        # Flatten heads for backward calc matching the logic
        # x_heads is [H, B, D/H]
        heads, batch, head_dim = x_heads.size(0), x_heads.size(1), x_heads.size(2)
        dim = heads * head_dim
        
        # [H, B, D/H] -> [B, H, D/H] -> [B, D]
        x_cat = x_heads.permute(1, 0, 2).contiguous().view(batch, dim)
        v_cat = v_heads.permute(1, 0, 2).contiguous().view(batch, dim)
        
        # 1. Gradients for V projection (Linear)
        # v_out = v_cat @ W_v^T
        # grad_W_v = grad_v_out.t().matmul(v_cat)
        # grad_v_cat = grad_v_out @ W_v
        grad_W_v = grad_v_out.t().matmul(v_cat)
        grad_v_cat_proj = grad_v_out.matmul(W_v)
        
        # 2. Gradients for X projection
        grad_W_x = None
        grad_x_cat = None
        grad_v_cat_mix = None
        
        if topology == 1: # Torus
             # Features: [sin(x), cos(x), tanh(v/100)]
             v_scaled = v_cat / 100.0
             v_mix = torch.tanh(v_scaled)
             s = torch.sin(x_cat)
             c = torch.cos(x_cat)
             
             features = torch.cat([s, c, v_mix], dim=-1) # [B, 3D]
             
             # x_out = features @ W_x^T
             grad_W_x = grad_x_out.t().matmul(features)
             grad_features = grad_x_out.matmul(W_x) # [B, 3D]
             
             # Split grad_features
             g_s = grad_features[:, :dim]
             g_c = grad_features[:, dim:2*dim]
             g_vm = grad_features[:, 2*dim:]
             
             # Backprop through sin/cos
             # d(sin)/dx = cos, d(cos)/dx = -sin
             grad_x_cat = g_s * c - g_c * s
             
             # Backprop through tanh(v/100)
             # d(tanh)/dv = (1-tanh^2)/100
             grad_v_cat_mix = g_vm * (1 - v_mix.pow(2)) / 100.0
             
        else: # Euclidean
             # x_out = x_cat @ W_x^T
             grad_W_x = grad_x_out.t().matmul(x_cat)
             grad_x_cat = grad_x_out.matmul(W_x)
             grad_v_cat_mix = torch.zeros_like(v_cat)

        # Combine v gradients
        grad_v_cat = grad_v_cat_proj + grad_v_cat_mix
        
        # Reshape back to heads [B, D] -> [H, B, D/H]
        # x_cat was [B, H*D_h]. x_heads was [H, B, D_h]
        # view(batch, heads, head_dim).permute(1, 0, 2)
        grad_x_heads = grad_x_cat.view(batch, heads, head_dim).permute(1, 0, 2).contiguous()
        grad_v_heads = grad_v_cat.view(batch, heads, head_dim).permute(1, 0, 2).contiguous()
        
        return grad_x_heads, grad_v_heads, grad_W_x, grad_W_v, None

def head_mixing_fused(x_heads, v_heads, W_x, W_v, topology=0):
    return HeadMixingFused.apply(x_heads, v_heads, W_x, W_v, topology)

def recurrent_manifold_fused(x, v, f, U_stack, W_stack, dt, dt_scales, forget_rates, num_heads, mix_x=None, mix_v=None, W_forget_stack=None, W_input_stack=None, b_forget_stack=None, W_potential_stack=None, b_potential_stack=None, topology=0, R=2.0, r=1.0, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0):
    """
    Fused recurrent manifold step with autograd support.
    """
    if CUDA_AVAILABLE and x.is_cuda:
        # Handle optional tensors
        mix_x = mix_x if mix_x is not None else torch.empty(0, device=x.device)
        mix_v = mix_v if mix_v is not None else torch.empty(0, device=x.device)
        W_forget_stack = W_forget_stack if W_forget_stack is not None else torch.empty(0, device=x.device)
        W_input_stack = W_input_stack if W_input_stack is not None else torch.empty(0, device=x.device)
        b_forget_stack = b_forget_stack if b_forget_stack is not None else torch.empty(0, device=x.device)
        W_potential_stack = W_potential_stack if W_potential_stack is not None else torch.empty(0, device=x.device)
        b_potential_stack = b_potential_stack if b_potential_stack is not None else torch.empty(0, device=x.device)
        
        return RecurrentManifoldFused.apply(
            x, v, f, U_stack, W_stack, dt, dt_scales, forget_rates, num_heads,
            mix_x, mix_v,
            W_forget_stack, W_input_stack, b_forget_stack,
            W_potential_stack, b_potential_stack,
            topology, R, r, plasticity, sing_thresh, sing_strength
        )
    else:
        # Python fallback not implemented
        return None

class RecurrentManifoldFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, v, f, U_stack, W_stack, dt, dt_scales, forget_rates, num_heads, mix_x, mix_v, W_forget_stack, W_input_stack, b_forget_stack, W_potential_stack, b_potential_stack, topology, R, r, plasticity, sing_thresh, sing_strength):
        # Clone inputs because kernel modifies them in-place
        x_in = x.clone()
        v_in = v.clone()
        
        ctx.dt = dt
        ctx.num_heads = num_heads
        ctx.topology = topology
        ctx.R = R
        ctx.r = r
        ctx.plasticity = plasticity
        ctx.sing_thresh = sing_thresh
        ctx.sing_strength = sing_strength
        
        outputs = gfn_cuda.recurrent_manifold_fused(
            x_in, v_in, f, U_stack, W_stack, dt, dt_scales, forget_rates, num_heads,
            plasticity, sing_thresh, sing_strength,
            mix_x, mix_v,
            W_forget_stack, W_input_stack, b_forget_stack,
            W_potential_stack, b_potential_stack,
            topology, R, r
        )
        
        x_state, v_state, x_out_seq, v_out_seq, reg_loss = outputs
        
        ctx.save_for_backward(x, v, f, U_stack, W_stack, dt_scales, forget_rates, mix_x, mix_v, W_forget_stack, W_input_stack, b_forget_stack, W_potential_stack, b_potential_stack, x_out_seq, v_out_seq)
        
        return x_out_seq, v_out_seq, x_state, v_state, reg_loss

    @staticmethod
    def backward(ctx, grad_x_seq, grad_v_seq, grad_x_final, grad_v_final, grad_reg_loss):
        x, v, f, U_stack, W_stack, dt_scales, forget_rates, mix_x, mix_v, W_forget_stack, W_input_stack, b_forget_stack, W_potential_stack, b_potential_stack, x_out_seq, v_out_seq = ctx.saved_tensors
        
        # Ensure gradients are contiguous
        grad_x_seq = grad_x_seq.contiguous() if grad_x_seq is not None else torch.zeros_like(x_out_seq)
        grad_v_seq = grad_v_seq.contiguous() if grad_v_seq is not None else torch.zeros_like(v_out_seq)
        grad_x_final = grad_x_final.contiguous() if grad_x_final is not None else torch.zeros_like(x)
        grad_v_final = grad_v_final.contiguous() if grad_v_final is not None else torch.zeros_like(v)
        
        grads = gfn_cuda.recurrent_manifold_backward(
            grad_x_seq, grad_v_seq, grad_x_final, grad_v_final,
            x, v, x_out_seq, v_out_seq,
            f, U_stack, W_stack,
            ctx.dt, dt_scales, forget_rates, ctx.num_heads,
            ctx.plasticity, ctx.sing_thresh, ctx.sing_strength,
            mix_x, mix_v,
            W_forget_stack, W_input_stack, b_forget_stack,
            W_potential_stack, b_potential_stack,
            ctx.topology, ctx.R, ctx.r
        )
        
        (g_x0, g_v0, g_f, g_U, g_W, g_mx, g_mv, g_fr, g_wf, g_wi, g_bf, g_wp, g_bp, g_dt_scales) = grads
        
        return g_x0, g_v0, g_f, g_U, g_W, None, g_dt_scales, g_fr, None, g_mx, g_mv, g_wf, g_wi, g_bf, g_wp, g_bp, None, None, None, None, None, None

