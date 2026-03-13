import torch
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gfn

def compute_acc(x_pred, targets_class):
    PI = math.pi
    TWO_PI = 2.0 * PI
    half_pi = PI * 0.5
    dist_pos = torch.min(torch.abs(x_pred - half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred - half_pi) % TWO_PI))
    dist_neg = torch.min(torch.abs(x_pred + half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred + half_pi) % TWO_PI))
    preds = (dist_pos.mean(dim=-1) < dist_neg.mean(dim=-1)).long()
    return (preds == targets_class).float().mean().item()

def diagnose():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # EXACT PRODUCTION CONFIG FROM logic_xor.py
    CONFIG = {
        'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16, 'impulse_scale': 80.0},
        'readout': {'type': 'implicit', 'coord_dim': 16},
        'active_inference': {
            'enabled': True,
            'dynamic_time': {'enabled': True},
            'reactive_curvature': {'enabled': True, 'plasticity': 0.2},
            'singularities': {'enabled': True, 'strength': 20.0, 'threshold': 0.8},
        },
        'fractal': {'enabled': True, 'threshold': 0.5, 'alpha': 0.2},
        'topology': {'type': 'torus'},
        'stability': {
            'enable_trace_normalization': True,
            'base_dt': 0.4,
            'velocity_saturation': 10.0,
            'friction': 0.05,
            'toroidal_curvature_scale': 0.01
        }
    }

    model = gfn.create(
        preset_name='stable-torus',
        vocab_size=2,
        dim=32,
        depth=2,
        heads=2,
        trajectory_mode='partition',
        holographic=True,
    ).to(device)

    max_steps = 300
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'impulse_scale' not in n], 'lr': 1e-3},
        {'params': [p for n, p in model.named_parameters() if 'impulse_scale' in n], 'lr': 2e-3},
    ])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, total_steps=max_steps, pct_start=0.2
    )
    
    criterion = gfn.loss('toroidal')
    
    print(f"Starting SYNCED XOR Diagnosis on {device} (with Active Inference + OneCycle)")
    
    for step in range(max_steps):
        x_in = torch.randint(0, 2, (128, 20), device=device)
        y_int = torch.cumsum(x_in, dim=1) % 2
        y_angle = (y_int.float() * 2.0 - 1.0) * (math.pi * 0.5)

        optimizer.zero_grad()
        output = model(x_in)
        x_pred = output[0]
        v_seq = output[2]['v_seq']
        
        y_expanded = y_angle.unsqueeze(-1).expand_as(x_pred)
        loss = criterion(x_pred, y_expanded)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Telemetry
        with torch.no_grad():
            v_mag = v_seq.norm(dim=-1).mean().item()
            acc = compute_acc(x_pred[:, -1, :], y_int[:, -1])
            curr_lr = scheduler.get_last_lr()[0]
            
            # Gradient telemetry
            grad_x0 = model.x0.grad.abs().max().item() if model.x0.grad is not None else 0.0
            grad_emb = list(model.embedding.parameters())[0].grad.abs().max().item() if list(model.embedding.parameters())[0].grad is not None else 0.0
            
        if step % 5 == 0 or step > 140:
            print(f"Step {step:03d} | Loss: {loss.item():.4f} | Avg Vel: {v_mag:.4f} | Acc: {acc*100:.1f}% | grad_x0: {grad_x0:.4e} | grad_emb: {grad_emb:.4e}")
            
        if step > 150 and loss.item() > 2.5:
            print("!!! COLLAPSE DETECTED !!!")
            break

if __name__ == "__main__":
    diagnose()
