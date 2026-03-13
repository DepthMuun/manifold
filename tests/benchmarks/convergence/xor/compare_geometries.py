import argparse
import copy
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import gfn
from tests.benchmarks.convergence.xor.logic_xor import PRODUCTION_PHYSICS_CONFIG, compute_accuracy


def build_config(base, topology_type=None, riemannian_type=None):
    cfg = copy.deepcopy(base)
    if topology_type is not None:
        cfg["topology"]["type"] = topology_type
    if riemannian_type is not None:
        cfg["topology"]["riemannian_type"] = riemannian_type
    return cfg


def run_case(name, physics, steps, batch_size, device):
    model = gfn.create(
        vocab_size=2,
        dim=8,
        depth=1,
        heads=2,
        physics=physics,
        trajectory_mode="partition",
        coupler_mode="mean_field",
        initial_spread=0.01,
        integrator="leapfrog",
        holographic=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = gfn.loss("toroidal")
    losses = []
    accs = []
    best_acc = 0.0

    for _ in range(steps):
        x_in = torch.randint(0, 2, (batch_size, 20), device=device)
        y_int = torch.cumsum(x_in, dim=1) % 2
        y_angle = (y_int.float() * 2.0 - 1.0) * (math.pi * 0.5)

        optimizer.zero_grad()
        output = model(x_in)[0]
        if output.dim() == 4:
            y_expanded = y_angle.unsqueeze(-1).unsqueeze(-1).expand_as(output)
        else:
            y_expanded = y_angle.unsqueeze(-1).expand_as(output)
        loss = criterion(output, y_expanded)
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            acc = compute_accuracy(output[:, -1, :], y_int[:, -1])
        losses.append(loss.item())
        accs.append(acc)
        if acc > best_acc:
            best_acc = acc

    avg_loss = sum(losses) / max(len(losses), 1)
    avg_acc = sum(accs) / max(len(accs), 1)

    return {
        "name": name,
        "avg_loss": avg_loss,
        "avg_acc": avg_acc,
        "best_acc": best_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cases = [
        ("torus", build_config(PRODUCTION_PHYSICS_CONFIG, topology_type="torus")),
        ("flat_torus", build_config(PRODUCTION_PHYSICS_CONFIG, topology_type="flat_torus")),
        ("low_rank", build_config(PRODUCTION_PHYSICS_CONFIG, topology_type="torus", riemannian_type="low_rank")),
        ("low_rank_paper", build_config(PRODUCTION_PHYSICS_CONFIG, topology_type="torus", riemannian_type="low_rank_paper")),
    ]

    results = []
    for name, cfg in cases:
        res = run_case(name, cfg, args.steps, args.batch_size, device)
        results.append(res)

    for res in results:
        print(
            f"{res['name']}: avg_loss={res['avg_loss']:.4f} | "
            f"avg_acc={res['avg_acc']:.3f} | best_acc={res['best_acc']:.3f}"
        )


if __name__ == "__main__":
    main()
