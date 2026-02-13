"""
Hyperparameter Optimization Runner
==================================

Orchestrates the search for optimal hyperparameters by:
1. Iterating through the configuration space.
2. Patching `gfn.constants` with trial values.
3. Running training tasks and consistency checks.
4. Logging results.
"""

import sys
import os
import json
import time
import importlib
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.optimization.config_space import get_grid_search_configs, get_random_search_configs, get_smoke_test_config
from tests.integration.test_overfit_sanity import run_training_task
import gfn.constants

def run_trial(config: Dict[str, Any], trial_id: int) -> Dict[str, Any]:
    """
    Runs a single optimization trial.
    
    Args:
        config: Hyperparameter configuration.
        trial_id: Identifier for this trial.
        
    Returns:
        Dictionary containing trial results and metrics.
    """
    print(f"\n--- Running Trial {trial_id} ---")
    print(f"Config: {config}")
    
    # 1. Patch constants
    # We need to construct a dictionary mapping 'gfn.constants.VAR' to value
    # But since we imported gfn.constants, we can use patch.object or patch.dict if it were a dict
    # Since constants are module attributes, we need to patch them on the module.
    
    patches = []
    for key, value in config.items():
        p = patch.object(gfn.constants, key, value)
        patches.append(p)
        p.start()
        
    try:
        # 2. Run Training Task
        print(">> Executing Training Task...")
        # Reloading modules might be necessary if they cache constants at import time
        # For now, we assume models read constants at runtime or construction
        # Note: Default args in function definitions are evaluated at import time! 
        # So we must ensure standard modules don't bake in constants in default args.
        
        train_metrics = run_training_task(config=None) # Config already applied via global patch
        
        # 3. Future: Run Consistency Check (Optional for speed, maybe only on best candidates)
        # For now, we skip explicit consistency check in the loop to save time, 
        # relying on training stability as a proxy.
        
        result = {
            "trial_id": trial_id,
            "status": "COMPLETED" if train_metrics["success"] else "FAILED",
            **config,
            **train_metrics
        }
        
    except Exception as e:
        print(f"!!! Trial Failed: {e}")
        result = {
            "trial_id": trial_id,
            "status": "ERROR",
            "error": str(e),
            **config
        }
    finally:
        # Stop all patches
        for p in patches:
            p.stop()
            
    return result

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization Runner")
    parser.add_argument("--mode", choices=["grid", "random", "smoke"], default="smoke", help="Search strategy")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples for random search")
    parser.add_argument("--output", type=str, default="optimization_results.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    print(f"Starting Optimization Run (Mode: {args.mode})")
    
    if args.mode == "grid":
        configs = get_grid_search_configs()
    elif args.mode == "random":
        configs = get_random_search_configs(args.samples)
    else:
        configs = get_smoke_test_config()
        
    print(f"Total configurations to test: {len(configs)}")
    
    results = []
    
    start_time = time.time()
    
    for i, config in enumerate(configs):
        trial_result = run_trial(config, i)
        results.append(trial_result)
        
        # Incremental save
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"Saved progress to {args.output}")
        
    total_time = time.time() - start_time
    print(f"\nOptimization Complete in {total_time:.2f}s")
    
    # Simple Analysis
    if results:
        df = pd.DataFrame(results)
        # Filter successful runs
        success_df = df[df["status"] == "COMPLETED"]
        
        if not success_df.empty:
            best_run = success_df.sort_values(by="final_loss").iloc[0]
            print("\n🏆 BEST CONFIGURATION:")
            print(best_run)
            
            # Save best config separate
            best_config = best_run[configs[0].keys()].to_dict()
            with open("best_hyperparameters.json", "w") as f:
                json.dump(best_config, f, indent=4)
                print("Saved best_hyperparameters.json")
        else:
            print("\n❌ No successful runs found.")

if __name__ == "__main__":
    main()
