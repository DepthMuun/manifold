"""
Hyperparameter Search Space Configuration
=========================================

Defines the ranges and distributions for hyperparameter optimization.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class HyperparameterConfig:
    name: str
    values: List[Any]
    description: str

# Define the search space
SEARCH_SPACE = [
    HyperparameterConfig(
        name="DEFAULT_LR",
        values=[1e-4, 5e-4, 1e-3, 5e-3],
        description="Learning rate"
    ),
    HyperparameterConfig(
        name="EMBEDDING_SCALE",
        values=[1.0, 1.5, 2.0],
        description="Scale for input embeddings"
    ),
    HyperparameterConfig(
        name="READOUT_GAIN",
        values=[1.0, 2.0, 5.0],
        description="Gain for the final readout layer"
    ),
    HyperparameterConfig(
        name="FRICTION_SCALE",
        values=[0.0, 0.02, 0.05, 0.1],
        description="Friction coefficient for symplectic integrators"
    ),
    HyperparameterConfig(
        name="DEFAULT_DT",
        values=[0.01, 0.02, 0.05, 0.1],
        description="Time step for integration"
    ),
     HyperparameterConfig(
        name="LEAPFROG_SUBSTEPS",
        values=[1, 3, 5],
        description="Number of substeps for Leapfrog integrator"
    ),
    HyperparameterConfig(
        name="LAMBDA_H_DEFAULT",
        values=[0.0, 0.001, 0.01],
        description="Hamiltonian regularization weight"
    )
]

def get_grid_search_configs() -> List[Dict[str, Any]]:
    """Generates a full grid of configurations."""
    import itertools
    
    keys = [c.name for c in SEARCH_SPACE]
    values = [c.values for c in SEARCH_SPACE]
    
    combinations = list(itertools.product(*values))
    configs = []
    
    for combo in combinations:
        config = dict(zip(keys, combo))
        configs.append(config)
        
    return configs

def get_random_search_configs(n_samples: int = 20) -> List[Dict[str, Any]]:
    """Generates random configurations from the search space."""
    import random
    
    configs = []
    for _ in range(n_samples):
        config = {}
        for param in SEARCH_SPACE:
            config[param.name] = random.choice(param.values)
        configs.append(config)
        
    return configs

def get_smoke_test_config() -> List[Dict[str, Any]]:
    """Minimal config for testing the runner."""
    return [
        {"DEFAULT_LR": 1e-4, "FRICTION_SCALE": 0.02},
        {"DEFAULT_LR": 1e-3, "FRICTION_SCALE": 0.0}
    ]
