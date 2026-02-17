import torch
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from gfn import Manifold
from gfn.losses import ToroidalDistanceLoss
from tests.benchmarks.viz.vis_gfn_superiority import ParityTask


def _standardize_forces(forces):
    m = forces.mean(dim=(0, 1), keepdim=True)
    s = forces.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    return (forces - m) / s


def test_initial_loss_within_tolerance():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    physics_config = {
        'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16},
        'readout': {'type': 'implicit', 'coord_dim': 16},
        'active_inference': {
            'enabled': True,
            'dynamic_time': {'enabled': True},
            'reactive_curvature': {'enabled': True, 'plasticity': 0.2},
            'singularities': {'enabled': True, 'strength': 2.0, 'threshold': 0.9}
        },
        'fractal': {'enabled': True, 'threshold': 0.5, 'alpha': 0.2},
        'topology': {'type': 'torus'},
        'stability': {'base_dt': 0.4},
        'cuda_fusion': {'allow_fused_training': False}
    }
    model = Manifold(
        vocab_size=2, dim=128, depth=6, heads=4,
        integrator_type='leapfrog', physics_config=physics_config,
        impulse_scale=80.0, holographic=True
    ).to(device)
    task = ParityTask(length=20)
    x, _, y_angle = task.generate_batch(64, device=device)
    with torch.no_grad():
        forces = model.embedding(x)
        forces = _standardize_forces(forces)
        out = model(input_ids=None, force_manual=forces, collect_christ=False)
        x_pred = out[0] if isinstance(out, tuple) else out
        y_expanded = y_angle.float().unsqueeze(-1).expand_as(x_pred)
        loss_fn = ToroidalDistanceLoss()
        loss = loss_fn(x_pred, y_expanded).item()
    assert abs(loss - 2.5) <= 0.25, f"Initial loss {loss:.2f} deviates more than 10% from 2.5"
