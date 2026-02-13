import torch
import torch.nn as nn
from ..geometry import (
    LowRankChristoffel, ReactiveChristoffel, HyperChristoffel, 
    EuclideanChristoffel, HyperbolicChristoffel, SphericalChristoffel,
    ToroidalChristoffel
)
from ..geometry.hierarchical import HierarchicalChristoffel
from ..geometry.adaptive import AdaptiveRankChristoffel
from ..integrators import (
    SymplecticIntegrator, RK4Integrator, HeunIntegrator, LeapfrogIntegrator, 
    YoshidaIntegrator, DormandPrinceIntegrator, EulerIntegrator,
    ForestRuthIntegrator, OmelyanIntegrator, CouplingFlowIntegrator, NeuralIntegrator
)
from .gating import RiemannianGating
from .thermo import ThermodynamicGating
from ..geometry.confusion import ConfusionChristoffel
from ..geometry.thermo import ThermodynamicChristoffel
from ..geometry.holographic import AdSCFTChristoffel
from ..geometry.ricci import RicciFlowChristoffel
from ..geometry.hysteresis import HysteresisChristoffel
from ..integrators.adaptive import AdaptiveIntegrator
from ..integrators.stochastic import StochasticIntegrator
from ..noise.geometric import GeometricNoise
from ..noise.curiosity import CuriosityNoise

class MLayer(nn.Module):
    """
    Manifold Layer (M-Layer):
    Takes current state (x, v) and input token force F.
    Evolves state via Geodesic Flow on multiple independent manifold subspaces.
    
    Architecture:
        1. Pre-LayerNorm (x, v)
        2. Split into K heads (Multi-Head Geodesic Flow)
        3. Parallel Geodesic Integration per head
        4. Concatenate & Mix
    
    Available integrators:
        - 'heun': Heun's method (RK2) [DEFAULT]
        - 'rk4': Runge-Kutta 4
        - 'rk45': Dormand-Prince (RK45)
        - 'symplectic': Velocity Verlet
        - 'leapfrog': Störmer-Verlet
    """
    def __init__(self, dim, heads=4, rank=16, base_dt=0.1, integrator_type='heun', physics_config=None, layer_idx=0, total_depth=6):
        super().__init__()
        assert dim % heads == 0, f"Dim {dim} must be divisible by heads {heads}"
        
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.physics_config = physics_config or {}
        self.base_dt = self.physics_config.get('stability', {}).get('base_dt', base_dt)
        
        self.layer_idx = layer_idx
        self.total_depth = total_depth
        self.depth_scale = 1.0 / (total_depth ** 0.5)
        
        self.norm_x = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        
        # Per-head manifold geometry with optional mixture and symmetries
        mixture_cfg = self.physics_config.get('mixture', {})
        mixture_enabled = mixture_cfg.get('enabled', False)
        
        head_rank = max(4, rank // heads)
        sym_cfg = self.physics_config.get('symmetries', {})
        isomeric_groups = sym_cfg.get('isomeric_groups', None) # e.g. [[0, 1], [2, 3]]
        
        self.christoffels = nn.ModuleList()
        christoffel_map = {}
        
        if isomeric_groups:
             # Symmetry groups share a single manifold instance
             pass

        def create_manifold(head_idx):
            topo_type = self.physics_config.get('topology', {}).get('type', 'euclidean').lower()
            is_torus = (topo_type == 'torus')

            if not mixture_enabled:
                 hyper = self.physics_config.get('hyper_curvature', {}).get('enabled', False)
                 hierarchical = self.physics_config.get('hierarchical_curvature', {}).get('enabled', False)
                 adaptive = self.physics_config.get('adaptive_curvature', {}).get('enabled', False)

                 if is_torus:
                      return ToroidalChristoffel(self.head_dim, physics_config=self.physics_config)
                 elif hierarchical:
                      ranks = self.physics_config.get('hierarchical_curvature', {}).get('ranks', [8, 16, 32])
                      return HierarchicalChristoffel(self.head_dim, ranks=ranks, physics_config=self.physics_config)
                 elif adaptive:
                      max_rank = self.physics_config.get('adaptive_curvature', {}).get('max_rank', 64)
                      return AdaptiveRankChristoffel(self.head_dim, max_rank=max_rank, physics_config=self.physics_config)
                 elif hyper:
                      return HyperChristoffel(self.head_dim, head_rank, physics_config=self.physics_config)
                 else:
                      return ReactiveChristoffel(self.head_dim, head_rank, physics_config=self.physics_config)
            
            # Mixture allocation: {'euclidean': [0], 'hyperbolic': [1], 'spherical': [2], 'learnable': [3]}
            comps = mixture_cfg.get('components', {})
            
            for type_name, indices in comps.items():
                if head_idx in indices:
                    if type_name == 'euclidean':
                        return EuclideanChristoffel(self.head_dim, physics_config=self.physics_config)
                    elif type_name == 'hyperbolic':
                        return HyperbolicChristoffel(self.head_dim, physics_config=self.physics_config)
                    elif type_name == 'spherical':
                        return SphericalChristoffel(self.head_dim, physics_config=self.physics_config)
                    elif type_name == 'learnable' or type_name == 'hyper':
                         return HyperChristoffel(self.head_dim, head_rank, physics_config=self.physics_config)
                    elif type_name == 'toroidal' or type_name == 'torus':
                         return ToroidalChristoffel(self.head_dim, physics_config=self.physics_config)
            
            return HyperChristoffel(self.head_dim, head_rank, physics_config=self.physics_config)

        for i in range(heads):
             if isomeric_groups:
                 found_group = False
                 for group in isomeric_groups:
                     if i in group:
                         if group[0] in christoffel_map:
                             christoffel_map[i] = christoffel_map[group[0]]
                         else:
                             instance = create_manifold(i)
                             christoffel_map[i] = instance
                             for member in group:
                                 christoffel_map[member] = instance
                         found_group = True
                         break
                 if found_group: continue
             
             christoffel_map[i] = create_manifold(i)
        
        for i in range(heads):
            # 0. Holographic Geometry (Paper 18) - Bulk Wrapper
            # This is the innermost wrapper as it defines the base bulk metric
            holo_cfg = self.physics_config.get('active_inference', {}).get('holographic_geometry', {})
            if holo_cfg.get('enabled', False):
                christoffel_map[i] = AdSCFTChristoffel(christoffel_map[i])

            # 1. Thermodynamic Geometry (Paper 15) - Inner Wrapper
            thermo_geo_cfg = self.physics_config.get('active_inference', {}).get('thermodynamic_geometry', {})
            if thermo_geo_cfg.get('enabled', False):
                temperature = thermo_geo_cfg.get('temperature', 1.0)
                alpha = thermo_geo_cfg.get('alpha', 0.1)
                christoffel_map[i] = ThermodynamicChristoffel(christoffel_map[i], temperature=temperature, alpha=alpha)

            # 2. Confusion Metric (Paper 08) - Plasticity Wrapper
            confusion_cfg = self.physics_config.get('active_inference', {}).get('confusion_metric', {})
            if confusion_cfg.get('enabled', False):
                sensitivity = confusion_cfg.get('sensitivity', 1.0)
                christoffel_map[i] = ConfusionChristoffel(christoffel_map[i], sensitivity=sensitivity)

            # 3. Ricci Flow (Paper 17) - Smoothing Wrapper (Outermost)
            ricci_cfg = self.physics_config.get('active_inference', {}).get('ricci_flow', {})
            if ricci_cfg.get('enabled', False):
                christoffel_map[i] = RicciFlowChristoffel(christoffel_map[i], lr=ricci_cfg.get('lr', 0.001))
            
            # 4. Hysteresis Memory (Paper 19) - Trajectory Deformation
            hyst_cfg = self.physics_config.get('hysteresis', {})
            if hyst_cfg.get('enabled', False):
                self.hysteresis_enabled = True
                christoffel_map[i] = HysteresisChristoffel(christoffel_map[i], self.head_dim, rank=hyst_cfg.get('rank', 16))
            else:
                self.hysteresis_enabled = False
                
            self.christoffels.append(christoffel_map[i])
            
        self.register_buffer('headless_mode', torch.tensor(False)) 

        self.use_dynamic_time = self.physics_config.get('active_inference', {}).get('dynamic_time', {}).get('enabled', False)
        
        topo_type = self.physics_config.get('topology', {}).get('type', 'euclidean').lower()
        self.topology_id = 1 if topo_type == 'torus' else 0

        if self.use_dynamic_time:
            gating_type = self.physics_config.get('active_inference', {}).get('dynamic_time', {}).get('type', 'learned')
            if gating_type == 'thermo':
                self.gatings = nn.ModuleList([
                    ThermodynamicGating(self.head_dim) for _ in range(heads)
                ])
            else:
                self.gatings = nn.ModuleList([
                    RiemannianGating(self.head_dim, topology=self.topology_id) for _ in range(heads)
                ])
        else:
             self.gatings = nn.ModuleList([
                RiemannianGating(self.head_dim, topology=self.topology_id) for _ in range(heads)
            ])
        
        # Initialize per-head timestep parameters
        # dt_params are initialized to give base_dt when softplus is applied:
        # dt = softplus(dt_param) where softplus(x) = log(1 + exp(x))
        # Initial value: target_dt = base_dt / 0.9, so dt ≈ base_dt
        # Each head gets a small offset (0.1 * i) for diversity
        param_iter = list(self.parameters())
        device = param_iter[0].device if len(param_iter) > 0 else torch.device('cpu')
        scale_vals = []
        for i in range(heads):
            target_dt = self.base_dt / 0.9
            val_init = torch.tensor(target_dt, device=device).exp().sub(1.0).log()
            val = val_init + i * 0.1
            scale_vals.append(val)
        self.dt_params = nn.Parameter(torch.stack(scale_vals))
        self.time_heads = None

        self.friction_gates = nn.ModuleList()
        
        for i in range(heads):
            if hasattr(self.christoffels[i], 'forget_gate'):
                 gate = self.christoffels[i].forget_gate
            else:
                 gate_in_dim = (3 if self.topology_id == 1 else 2) * self.head_dim
                 gate = nn.Linear(gate_in_dim, self.head_dim) 
                 nn.init.orthogonal_(gate.weight, gain=0.5)
                 nn.init.constant_(gate.bias, 0.0) 
            
            self.friction_gates.append(gate)

        self.integrators = nn.ModuleList()
        for i in range(heads):
            if integrator_type == 'rk4':
                integ = RK4Integrator(self.christoffels[i], dt=self.base_dt)
            elif integrator_type == 'rk45':
                integ = DormandPrinceIntegrator(self.christoffels[i], dt=self.base_dt)
            elif integrator_type == 'heun':
                integ = HeunIntegrator(self.christoffels[i], dt=self.base_dt)
            elif integrator_type == 'euler':
                integ = EulerIntegrator(self.christoffels[i], dt=self.base_dt)
            elif integrator_type == 'leapfrog':
                integ = LeapfrogIntegrator(self.christoffels[i], dt=self.base_dt)
            elif integrator_type == 'yoshida':
                integ = YoshidaIntegrator(self.christoffels[i], dt=self.base_dt)
            elif integrator_type == 'forest_ruth':
                integ = ForestRuthIntegrator(self.christoffels[i], dt=self.base_dt)
            elif integrator_type == 'omelyan':
                integ = OmelyanIntegrator(self.christoffels[i], dt=self.base_dt)
            elif integrator_type == 'pefrl':
                integ = PEFRLIntegrator(self.christoffels[i], dt=self.base_dt)
            elif integrator_type == 'coupling':
                integ = CouplingFlowIntegrator(self.christoffels[i], dt=self.base_dt)
            elif integrator_type == 'neural':
                integ = NeuralIntegrator(self.christoffels[i], dt=self.base_dt, dim=self.head_dim)
            else:
                integ = SymplecticIntegrator(self.christoffels[i], dt=self.base_dt)
            self.integrators.append(integ)
            
        # Wrap with Adaptive Integrator if enabled (Paper 07)
        adaptive_cfg = self.physics_config.get('active_inference', {}).get('adaptive_resolution', {})
        if adaptive_cfg.get('enabled', False):
            tolerance = adaptive_cfg.get('tolerance', 1e-3)
            max_depth = adaptive_cfg.get('max_depth', 3)
            for i in range(heads):
                self.integrators[i] = AdaptiveIntegrator(
                    self.integrators[i], 
                    tolerance=tolerance, 
                    max_depth=max_depth
                )

        # 4. Curiosity Noise (Paper 26)
        curiosity_cfg = self.physics_config.get('active_inference', {}).get('curiosity_noise', {})
        if curiosity_cfg.get('enabled', False):
            self.curiosity_noises = nn.ModuleList([
                CuriosityNoise(
                    self.head_dim, 
                    base_std=curiosity_cfg.get('base_std', 0.01),
                    sensitivity=curiosity_cfg.get('sensitivity', 1.0)
                ) for _ in range(heads)
            ])
        else:
            self.curiosity_noises = None
            
        # Wrap with Stochastic Integrator if enabled (Paper 16)
        stochastic_cfg = self.physics_config.get('active_inference', {}).get('stochastic_geometry', {})
        if stochastic_cfg.get('enabled', False):
            # We use a ModuleList to store noise modules so they are on same device
            self.noises = nn.ModuleList([
                GeometricNoise(self.head_dim, sigma=stochastic_cfg.get('sigma', 0.01))
                for _ in range(heads)
            ])
            for i in range(heads):
                self.integrators[i] = StochasticIntegrator(
                    self.integrators[i],
                    self.noises[i]
                )
            
        if heads > 1:
            self.out_proj_x = nn.Linear(3 * dim if self.topology_id == 1 else dim, dim)
            self.out_proj_v = nn.Linear(dim, dim)
            
            self.mixed_norm_x = nn.RMSNorm(dim)
            self.mixed_norm_v = nn.RMSNorm(dim)
            
            nn.init.xavier_uniform_(self.out_proj_x.weight)
            nn.init.zeros_(self.out_proj_x.bias)
            nn.init.xavier_uniform_(self.out_proj_v.weight)
            nn.init.zeros_(self.out_proj_v.bias)
            
        self.use_recursive = self.physics_config.get('active_inference', {}).get('recursive_geodesics', {}).get('enabled', False)
        if self.use_recursive:
            self.context_proj = nn.Linear(heads, dim)
            nn.init.zeros_(self.context_proj.weight)
            
    def forward(self, x, v, force=None, context=None, collect_christ=False, memory_state=None):
        """
        Args:
            x: Position [Batch, Dim]
            v: Velocity [Batch, Dim]
            force: Input force F [Batch, Dim]
            context: Optional previous scale context
            collect_christ: Whether to return Christoffel outputs
            memory_state: Accumulated hysteresis state [Batch, Dim]
        """
        batch = x.shape[0]
        
        # 1. Split into Heads
        x_heads = x.view(batch, self.heads, self.head_dim).unbind(dim=1)
        v_heads = v.view(batch, self.heads, self.head_dim).unbind(dim=1)
        
        if memory_state is not None:
            m_heads = memory_state.view(batch, self.heads, self.head_dim).unbind(dim=1)
        else:
            m_heads = [None] * self.heads

        if force is not None:
             if self.use_recursive and context is not None:
                 force = force + self.context_proj(context)
             f_heads = force.view(batch, self.heads, self.head_dim).unbind(dim=1)
        else:
             f_heads = [None] * self.heads
             
        # 3. Vectorized Gating [Heads, Batch, 1]
        dt_base = torch.nn.functional.softplus(self.dt_params).view(self.heads, 1, 1)
        
        # AUDIT FIX: Clamp dt_scale to reasonable range
        # Prevents extreme timesteps that cause numerical instability
        # Range [0.1, 2.0] provides stable integration across most scenarios
        stability_cfg = self.physics_config.get('stability', {})
        dt_min = stability_cfg.get('dt_min', self.base_dt * 0.1)
        dt_max = stability_cfg.get('dt_max', self.base_dt * 4.0)
        dt_base = torch.clamp(dt_base, dt_min, dt_max)
        
        # Only apply dynamic gating if enabled
        use_gating = self.physics_config.get('active_inference', {}).get('dynamic_time', {}).get('enabled', False)
        
        if use_gating:
            # Check integration signature: Thermodynamic needs v, Riemmanian only x
            gates_list = []
            for i in range(self.heads):
                gate_mod = self.gatings[i]
                if isinstance(gate_mod, ThermodynamicGating):
                    gates_list.append(gate_mod(x_heads[i], v_heads[i]))
                else:
                    gates_list.append(gate_mod(x_heads[i]))
            
            gates = torch.stack(gates_list, dim=0)
            scale = dt_base * gates # [Heads, Batch, 1]
        else:
            # Dummy gates for context passing
            gates = torch.ones(self.heads, batch, 1, device=x.device, dtype=x.dtype)
            scale = dt_base # Static scaling matching CUDA kernel default
        
        # Clutch parameter stacking for the kernel
        
        W_f_list = []
        W_i_list = []
        b_f_list = []
        
        for i in range(self.heads):
            # weight: [Out, In]
            head_geo = self.christoffels[i]
            
            if hasattr(head_geo, 'forget_gate') and hasattr(head_geo, 'input_gate'):
                # Separate state and force gates
                W_f_list.append(head_geo.forget_gate.weight)
                W_i_list.append(head_geo.input_gate.weight)
                b_f_list.append(head_geo.forget_gate.bias)
            else:
                # Legacy/combined gate
                w = self.friction_gates[i].weight
                b = self.friction_gates[i].bias
                d = self.head_dim
                
                if self.topology_id == 1:
                    # W_forget = [D, 2D] (sin, cos)
                    # W_input = [D, D] (force)
                    W_f_list.append(w[:, :2*d])
                    W_i_list.append(w[:, 2*d:])
                else:
                    W_f_list.append(w[:, :d])
                    W_i_list.append(w[:, d:])
                b_f_list.append(b)
        
        # W_forget_stack is [H, D, 2D] for torus
        W_forget_stack = torch.stack(W_f_list, dim=0).contiguous()
        W_input_stack = torch.stack(W_i_list, dim=0).contiguous() 
        b_forget_stack = torch.stack(b_f_list, dim=0).contiguous()

        # Batched geodesic step
        x_outs = []
        v_outs = []
        christoffel_outputs = []
        
        # Legacy per-head call with clutch weights
        
        for i in range(self.heads):
            # Per-head step with clutch weights
            
            extra_kwargs = {
                'W_forget_stack': W_forget_stack[i:i+1], # [1, D, D]
                'W_input_stack': W_input_stack[i:i+1],
                'b_forget_stack': b_forget_stack[i:i+1],
                'topology': self.topology_id,
                'collect_christ': collect_christ,
                'memory_state': m_heads[i]
            }
            
            res = self.integrators[i](x_heads[i], v_heads[i], force=f_heads[i], dt_scale=scale[i], **extra_kwargs)
            
            if collect_christ:
                xh, vh, ch_out = res
                christoffel_outputs.append(ch_out)
            else:
                xh, vh = res
            
            # --- Proactive Curiosity Exploration (Paper 26) ---
            if self.curiosity_noises is not None:
                vh = self.curiosity_noises[i](vh, force=f_heads[i], training=self.training)

            x_outs.append(xh)
            v_outs.append(vh)

        # 6. Concatenate and Mix (Standard)
        if self.heads > 1 and not collect_christ:
            try:
                from gfn.cuda.ops import head_mixing_fused, CUDA_AVAILABLE
                if CUDA_AVAILABLE and x.is_cuda:
                    x_stacked = torch.stack(x_outs, dim=0)
                    v_stacked = torch.stack(v_outs, dim=0)
                    x_next, v_next = head_mixing_fused(x_stacked, v_stacked, self.out_proj_x.weight, self.out_proj_v.weight)
                    context_next = gates.squeeze(-1).transpose(0, 1)
                    return x_next, v_next, context_next, christoffel_outputs
            except: pass
        
        # 6. Concatenate and Mix
        x_cat = torch.stack(x_outs, dim=1).view(batch, -1)
        v_cat = torch.stack(v_outs, dim=1).view(batch, -1)
        
        if self.heads > 1:
            if self.topology_id == 1:
                 # PERIODIC MIXING: Mixer sees [sin(x), cos(x), v]
                 v_mix = torch.tanh(v_cat / 100.0)
                 mixer_in_x = torch.cat([torch.sin(x_cat), torch.cos(x_cat), v_mix], dim=-1)
                 x_next = self.out_proj_x(mixer_in_x)
            else:
                 x_next = self.out_proj_x(x_cat)
            
            v_next = self.out_proj_v(v_cat)
            
            # Normalize to prevent magnitude creep (Bypass for Torus to preserve phase)
            # Normalize to prevent magnitude creep (Bypass for Torus to preserve phase)
            if self.topology_id != 1:
                x_next = self.mixed_norm_x(x_next)
            else:
              
                # AUDIT FIX: Project back to [-π, π] with smooth wrapping
                # Use atan2 for differentiable gradients
                x_next = torch.atan2(torch.sin(x_next), torch.cos(x_next))
                
            v_next = self.mixed_norm_v(v_next)
        else:
            x_next, v_next = x_cat, v_cat
            
        # Velocity Saturation (Relativistic Bounding)
        v_next = 100.0 * torch.tanh(v_next / 100.0)
            
        context_next = gates.squeeze(-1).transpose(0, 1)
        return x_next, v_next, context_next, christoffel_outputs
