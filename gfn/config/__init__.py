"""
gfn/config/__init__.py
Public API for the configuration module — GFN V5
Centralized configuration system for all GFN components.
"""

from gfn.config.schema import (
    TopologyConfig,
    StabilityConfig,
    DynamicTimeConfig,
    HysteresisConfig,
    ActiveInferenceConfig,
    EmbeddingConfig,
    ReadoutConfig,
    MixtureConfig,
    DynamicsConfig,
    FractalConfig,
    SingularityConfig,
    PhysicsConfig,
    TrainerConfig,
    ManifoldConfig,
)

from gfn.config.defaults import (
    PHYSICS_DEFAULTS,
    MODEL_DEFAULTS,
    TRAINING_DEFAULTS,
    LOSS_DEFAULTS,
    get_default,
)

from gfn.config.loader import dict_to_physics_config
from gfn.config.validator import ConfigValidator, validate_manifold_config, validate_and_print

__all__ = [
    # Schema classes
    "TopologyConfig",
    "StabilityConfig", 
    "DynamicTimeConfig",
    "HysteresisConfig",
    "ActiveInferenceConfig",
    "EmbeddingConfig",
    "ReadoutConfig",
    "MixtureConfig",
    "DynamicsConfig",
    "FractalConfig",
    "SingularityConfig",
    "PhysicsConfig",
    "TrainerConfig",
    "ManifoldConfig",
    # Defaults
    "PHYSICS_DEFAULTS",
    "MODEL_DEFAULTS",
    "TRAINING_DEFAULTS",
    "LOSS_DEFAULTS",
    "get_default",
    # Loader
    "dict_to_physics_config",
    # Validator
    "ConfigValidator",
    "validate_manifold_config",
    "validate_and_print",
]
