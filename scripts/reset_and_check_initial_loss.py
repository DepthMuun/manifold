import torch
import torch.nn as nn
import argparse
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from gfn import Manifold
from gfn.losses import ToroidalDistanceLoss
from tests.benchmarks.viz.vis_gfn_superiority import ParityTask, _standardize_forces


def reinit_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

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

    model.apply(reinit_xavier)

    task = ParityTask(length=20)
    x, y_class, y_angle = task.generate_batch(128, device=device)
    with torch.no_grad():
        forces = model.embedding(x)
        forces = _standardize_forces(forces)
        out = model(input_ids=None, force_manual=forces, collect_christ=False)
        x_pred = out[0] if isinstance(out, tuple) else out
        y_expanded = y_angle.float().unsqueeze(-1).expand_as(x_pred)
        loss_fn = ToroidalDistanceLoss()
        loss = loss_fn(x_pred, y_expanded).item()
        baseline = (torch.abs(torch.atan2(torch.sin(-y_angle), torch.cos(-y_angle)))**2).mean().item()
        print(f"[BASELINE] Target = {baseline:.2f}")
        print(f"[CHECK] Initial batch loss = {loss:.2f}")
        ok = abs(loss - 2.5) <= 0.25
        print(f"[RESULT] {'OK' if ok else 'DRIFT'}")


if __name__ == "__main__":
    main()
