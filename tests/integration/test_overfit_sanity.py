import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold
from gfn.losses import GFNLoss
from gfn.optim import RiemannianAdam

def run_training_task(config: dict = None, device_str: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Runs a training task with the given configuration and returns performance metrics.
    
    Args:
        config: Dictionary of hyperparameters to override in gfn.constants
        device_str: Device to run on ('cpu' or 'cuda')
        
    Returns:
        dict: Metrics including final_loss, steps_to_convergence, etc.
    """
    import time
    from unittest.mock import patch
    
    # Default config if None
    if config is None:
        config = {}
        
    # Apply configuration overrides using patch
    # We need to patch gfn.constants for the duration of this function
    # Note: This is tricky because constants are imported at module level in many places.
    # The runner.py will handle the patching at a higher level, so here we assume
    # constants are already set or passed into model constructor if possible.
    # Ideally, Manifold and other classes should accept these as args.
    # For now, we will rely on the caller to have patched constants OR
    # we can try to reload/patch here if we really want isolation.
    # Better approach for this task: The Manifold model takes some args, others are global.
    # We will assume global constants are patched by the caller for simplicity in this step.
    
    print(f"=== TRAINING TASK START ({device_str}) ===")
    device = torch.device(device_str)
    
    # Extract config or use defaults
    vocab_size = config.get('vocab_size', 16)
    dim = config.get('dim', 256)
    depth = config.get('depth', 12)
    rank = config.get('rank', 64)
    heads = config.get('heads', 4)
    lr = config.get('lr', 1e-3)
    lambda_h = config.get('lambda_h', 0.01)
    
    print(f"Initializing Model (dim={dim}, depth={depth}, heads={heads}, lr={lr})...")
    try:
        model = Manifold(vocab_size, dim, depth, rank, heads=heads, integrator_type='leapfrog').to(device)
        optimizer = RiemannianAdam(model.parameters(), lr=lr)
        criterion = GFNLoss(lambda_h=lambda_h)
        
        # Fixed Batch (Overfit Target)
        x = torch.tensor([[1, 10, 1, 13, 2, 15] for _ in range(4)]).to(device) # Batch 4
        targets = x.clone()
        
        print("Starting Training Loop (100 steps)...")
        model.train()
        
        initial_loss = None
        final_loss = None
        steps_to_convergence = 100
        diverged = False
        
        start_time = time.time()
        
        for i in range(100):
            optimizer.zero_grad()
            output = model(x)
            logits = output[0]
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            
            loss, _ = criterion(shift_logits, shift_targets)
            
            if torch.isnan(loss):
                print(f"❌ Divergence at step {i}")
                diverged = True
                final_loss = float('inf')
                break
                
            loss.backward()
            
            # Clip grad to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            current_loss = loss.item()
            if i == 0: initial_loss = current_loss
            
            if i % 10 == 0:
                print(f"Step {i}: Loss {current_loss:.4f}")
            
            final_loss = current_loss
            
            # Simple convergence check
            if current_loss < 0.01:
                steps_to_convergence = i
                break
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Final Loss: {final_loss:.4f}")
        
        success = final_loss < 0.1 and not diverged
        
        return {
            "success": success,
            "final_loss": final_loss,
            "initial_loss": initial_loss,
            "steps": steps_to_convergence,
            "duration": duration,
            "diverged": diverged
        }

    except Exception as e:
        print(f"💥 Optimization Failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "final_loss": float('inf')
        }

if __name__ == "__main__":
    # Test wrapper for standalone execution
    result = run_training_task()
    print("\nResult:", result)

