
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold
from gfn.optim import RiemannianAdam
from gfn.losses import geodesic_regularization, hamiltonian_loss, ToroidalDistanceLoss

# --- Configuration & Helpers ---

@dataclass
class ExperimentConfig:
    name: str
    loss_config: Dict[str, float]  # {'hamiltonian': 0.0, 'geodesic': 0.0}
    optimizer_type: str  # 'adamw' or 'riemannian'
    retraction: str = 'torus'  # for riemannian
    learning_rate: float = 5e-4
    steps: int = 200
    batch_size: int = 64
    dim: int = 64
    heads: int = 4
    depth: int = 2

class ParityTask:
    """Parity Check (Modulo 2) for state tracking."""
    def __init__(self, vocab_size=2, length=20, mod=2):
        self.vocab_size = vocab_size
        self.length = length
        self.mod = mod
        
    def generate_batch(self, batch_size, device='cpu'):
        x = torch.randint(0, self.vocab_size, (batch_size, self.length), device=device)
        y_int = torch.cumsum(x, dim=1) % self.mod
        PI = 3.14159265359
        y_angle = (y_int.float() * 2.0 - 1.0) * (PI * 0.5)
        return x, y_int, y_angle

def train_one_experiment(config: ExperimentConfig, device: str):
    print(f"\n[bold cyan]--- Running Experiment: {config.name} ---[/]")
    
    # 1. Model Setup
    physics_config = {
        'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16}, 
        'readout': {'type': 'implicit', 'coord_dim': 16},
        'active_inference': {
            'enabled': True, 
            'dynamic_time': {'enabled': False},
            'reactive_curvature': {'enabled': True, 'plasticity': 0.05},
            'singularities': {'enabled': True, 'strength': 3.0, 'threshold': 0.8}
        },
        'fractal': {'enabled': True, 'threshold': 0.5, 'alpha': 0.2},
        'topology': {'type': 'torus'},
        'stability': {'base_dt': 0.05}
    }
    
    model = Manifold(
        vocab_size=2, 
        dim=config.dim, 
        depth=config.depth, 
        heads=config.heads, 
        integrator_type='leapfrog', 
        physics_config=physics_config, 
        impulse_scale=10.0, 
        holographic=True
    ).to(device)
    
    # 2. Optimizer Setup
    if config.optimizer_type == 'riemannian':
        # Separate parameters
        torus_params = [p for n, p in model.named_parameters() if 'x0' in n]
        euclidean_params = [p for n, p in model.named_parameters() if any(x in n for x in ['v0', 'impulse_scale', 'gate']) and 'x0' not in n]
        standard_params = [p for n, p in model.named_parameters() if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])]
        
        optimizer = RiemannianAdam([
            {'params': standard_params, 'lr': config.learning_rate, 'weight_decay': 1e-4, 'retraction': 'normalize'},
            {'params': torus_params, 'lr': config.learning_rate, 'weight_decay': 0.0, 'retraction': config.retraction},
            {'params': euclidean_params, 'lr': config.learning_rate, 'weight_decay': 0.0, 'retraction': 'euclidean'}
        ])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

    # 3. Training Loop
    criterion_val = ToroidalDistanceLoss()
    history = {'loss': [], 'acc': [], 'grad_norm': []}
    
    start_time = time.time()
    
    task = ParityTask(length=20)
    
    for step in range(config.steps):
        x, y_class, y_angle = task.generate_batch(config.batch_size, device=device)
        
        optimizer.zero_grad()
        output = model(x, collect_christ=False)
        
        # Unpack output
        if isinstance(output, tuple):
            x_pred = output[0]
            christoffels = output[2]
            v_seq = output[3]
            x_seq = output[4]
            all_forces = output[5]
        else:
            x_pred = output
            christoffels, v_seq, x_seq, all_forces = None, None, None, None
            
        # Target expansion
        y_expanded = y_angle.float().unsqueeze(-1).expand_as(x_pred)
        
        # Loss Calculation
        loss_main = criterion_val(x_pred, y_expanded)
        loss_ham = 0.0
        loss_geo = 0.0
        
        if config.loss_config.get('hamiltonian', 0.0) > 0 and v_seq is not None:
             metric_fn = None
             if hasattr(model.layers[0], "christoffels") and model.layers[0].christoffels:
                 if hasattr(model.layers[0].christoffels[0], "get_metric"):
                     def first_head_metric(x):
                         return model.layers[0].christoffels[0].get_metric(x)
                     metric_fn = first_head_metric
             
             loss_ham = hamiltonian_loss(
                 v_seq, 
                 states=x_seq, 
                 metric_fn=metric_fn, 
                 lambda_h=config.loss_config['hamiltonian'], 
                 forces=all_forces
             )
             
        if config.loss_config.get('geodesic', 0.0) > 0 and christoffels is not None:
            loss_geo = geodesic_regularization(
                None, 
                christoffels, 
                lambda_g=config.loss_config['geodesic']
            )
            
        total_loss = loss_main + loss_ham + loss_geo
        
        if torch.isnan(total_loss):
            print(f"!! NaN Loss detected at step {step} !!")
            break
            
        total_loss.backward()
        
        # Gradient Norm
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        history['grad_norm'].append(total_norm.item())
        
        optimizer.step()
        
        # Accuracy Check
        with torch.no_grad():
            PI = 3.14159265359
            TWO_PI = 2.0 * PI
            half_pi = PI * 0.5
            dist_pos = torch.min(torch.abs(x_pred - half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred - half_pi) % TWO_PI))
            dist_neg = torch.min(torch.abs(x_pred + half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred + half_pi) % TWO_PI))
            d_pos = dist_pos.mean(dim=-1)
            d_neg = dist_neg.mean(dim=-1)
            preds = (d_pos < d_neg).long()
            acc = (preds == y_class).float().mean().item()
        
        history['loss'].append(total_loss.item())
        history['acc'].append(acc)
        
        if step % 50 == 0:
            print(f"Step {step}: Loss={total_loss.item():.4f} (H={loss_ham:.4f}, G={loss_geo:.4f}), Acc={acc*100:.1f}%, Grad={total_norm:.2f}")

    duration = time.time() - start_time
    final_acc = np.mean(history['acc'][-10:])
    final_loss = np.mean(history['loss'][-10:])
    
    print(f"Result: Final Acc={final_acc*100:.1f}%, Final Loss={final_loss:.4f}, Time={duration:.2f}s")
    return {
        'config': config.name,
        'final_acc': final_acc,
        'final_loss': final_loss,
        'history': history,
        'duration': duration
    }

def run_ablation_suite():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")
    
    experiments = [
        # 1. Baseline: No auxiliary losses, AdamW (Control)
        ExperimentConfig(
            name="AdamW_NoAux",
            loss_config={'hamiltonian': 0.0, 'geodesic': 0.0},
            optimizer_type='adamw'
        ),
        # 2. Riemannian: No auxiliary losses, RiemannianAdam (Torus)
        ExperimentConfig(
            name="Riemannian_NoAux",
            loss_config={'hamiltonian': 0.0, 'geodesic': 0.0},
            optimizer_type='riemannian'
        ),
        # 3. Hamiltonian Low: Riemannian + Small Hamiltonian
        ExperimentConfig(
            name="Riemannian_HamLow",
            loss_config={'hamiltonian': 1e-4, 'geodesic': 0.0},
            optimizer_type='riemannian'
        ),
        # 4. Hamiltonian High: Riemannian + High Hamiltonian
        ExperimentConfig(
            name="Riemannian_HamHigh",
            loss_config={'hamiltonian': 1e-2, 'geodesic': 0.0},
            optimizer_type='riemannian'
        ),
        # 5. Full: Riemannian + Hamiltonian + Geodesic
        ExperimentConfig(
            name="Riemannian_Full",
            loss_config={'hamiltonian': 1e-4, 'geodesic': 1e-3},
            optimizer_type='riemannian'
        ),
    ]
    
    results = []
    for exp in experiments:
        res = train_one_experiment(exp, device)
        results.append(res)
        torch.cuda.empty_cache()
        
    # Summary
    print("\n" + "="*60)
    print(" " * 20 + "ABLATION RESULTS")
    print("="*60)
    print(f"{'Experiment':<20} | {'Acc':<8} | {'Loss':<8} | {'Time':<8}")
    print("-" * 60)
    for res in results:
        print(f"{res['config']:<20} | {res['final_acc']*100:.1f}%   | {res['final_loss']:.4f}   | {res['duration']:.2f}s")
    print("="*60)

if __name__ == "__main__":
    from rich.console import Console
    console = Console()
    try:
        run_ablation_suite()
    except Exception as e:
        console.print_exception()
