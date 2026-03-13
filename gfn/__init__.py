"""
GFN (Geodesic Flow Network) V5
==============================
API pública del paquete — compatibilidad con benchmarks y código legado.
"""

# ── Modelos ───────────────────────────────────────────────────────────────────
from gfn.models.manifold import ManifoldModel as Manifold
from gfn.models.manifold import ManifoldModel as Model
from gfn.models.manifold_layer import ManifoldLayer, ManifoldLayer as MLayer

# ── Configuración ─────────────────────────────────────────────────────────────
from gfn.config.schema import (
    ManifoldConfig, PhysicsConfig, TopologyConfig, StabilityConfig,
    ActiveInferenceConfig, EmbeddingConfig, ReadoutConfig, DynamicTimeConfig,
    MixtureConfig, DynamicsConfig, FractalConfig, SingularityConfig, HysteresisConfig
)

# ── Integradores ──────────────────────────────────────────────────────────────
from gfn.physics.integrators.symplectic.leapfrog import LeapfrogIntegrator  as Leapfrog
from gfn.physics.integrators.runge_kutta.heun    import HeunIntegrator      as Heun
from gfn.physics.integrators.symplectic.yoshida  import YoshidaIntegrator   as Yoshida
from gfn.physics.integrators.symplectic.verlet   import VerletIntegrator    as Verlet
from gfn.physics.integrators.runge_kutta.rk4     import RK4Integrator       as RK4

# ── Geometría ─────────────────────────────────────────────────────────────────
from gfn.geometry.torus    import ToroidalRiemannianGeometry
from gfn.geometry.low_rank import LowRankRiemannianGeometry  as LowRankChristoffel
from gfn.geometry.reactive import ReactiveRiemannianGeometry as ReactiveChristoffel

# ── Física: Dynamics ──────────────────────────────────────────────────────────
from gfn.physics.dynamics import (
    get_dynamics, BaseDynamics,
    DirectDynamics, ResidualDynamics, MixDynamics, GatedDynamics, StochasticDynamics
)

# ── Física: Normalization y Gating ────────────────────────────────────────────
from gfn.physics.normalization import ManifoldNormalizationRegistry
from gfn.physics.gating import RiemannianGating, ThermodynamicLayer, FrictionGate
from gfn.physics.hamiltonian import HamiltonianTrajectorySolver
from gfn.physics.monitor import PhysicsMonitorPlugin

# ── Aggregators ───────────────────────────────────────────────────────────────
from gfn.models.components.pooling import (
    HamiltonianPooling, HierarchicalAggregator, MomentumAggregator
)

# ── Optimizadores ─────────────────────────────────────────────────────────────
from gfn.training.optimizer import RiemannianAdam, RiemannianSGD, ManifoldSGD

# ── Evaluación ────────────────────────────────────────────────────────────────
from gfn.training.evaluation import ManifoldMetricEvaluator, PhysicsConstraintEvaluator


# ── API Clean — create / loss / load / benchmark ──────────────────────────────
from gfn.api import create, loss, save, load, benchmark, Trainer

# ── Losses — Detection ────────────────────────────────────────────────────────
from gfn.losses.detection import GIoULoss, IoULoss, giou_loss, iou_loss

# ── Utils — Coords ────────────────────────────────────────────────────────────
from gfn.utils.coords import box_to_torus, torus_to_box, wrap_angles, angle_to_unit

# ── Training — Optimizer utils ────────────────────────────────────────────────
from gfn.training.optimizer import make_gfn_optimizer, all_parameters

# ── Training — Checkpoint ─────────────────────────────────────────────────────
from gfn.training.checkpoint import save_checkpoint, load_checkpoint


# ── __all__ ───────────────────────────────────────────────────────────────────
__all__ = [
    # Modelos
    'Manifold', 'Model', 'ManifoldLayer', 'MLayer',
    # Configuración
    'ManifoldConfig', 'PhysicsConfig', 'TopologyConfig', 'StabilityConfig',
    'ActiveInferenceConfig', 'EmbeddingConfig', 'ReadoutConfig', 'DynamicTimeConfig',
    'MixtureConfig', 'DynamicsConfig', 'FractalConfig', 'SingularityConfig', 'HysteresisConfig',
    # Integradores
    'Leapfrog', 'Heun', 'Yoshida', 'Verlet', 'RK4',
    # Geometría
    'ToroidalRiemannianGeometry', 'LowRankChristoffel', 'ReactiveChristoffel',
    # Dynamics
    'get_dynamics', 'BaseDynamics',
    'DirectDynamics', 'ResidualDynamics', 'MixDynamics', 'GatedDynamics', 'StochasticDynamics',
    # Física
    'ManifoldNormalizationRegistry',
    'RiemannianGating', 'ThermodynamicLayer', 'FrictionGate',
    'HamiltonianTrajectorySolver',
    'PhysicsMonitorPlugin',
    # Aggregators
    'HamiltonianPooling', 'HierarchicalAggregator', 'MomentumAggregator',
    # Optimizadores
    'RiemannianAdam',
    'make_gfn_optimizer', 'all_parameters',
    # Evaluación
    'ManifoldMetricEvaluator', 'PhysicsConstraintEvaluator',
    # Losses — Detection
    'GIoULoss', 'IoULoss', 'giou_loss', 'iou_loss',
    # Utils — Coords
    'box_to_torus', 'torus_to_box', 'wrap_angles', 'angle_to_unit',
    # Training — Checkpoint
    'save_checkpoint', 'load_checkpoint',
    # API Clean
    'create', 'loss', 'save', 'load', 'benchmark', 'Trainer',
]
