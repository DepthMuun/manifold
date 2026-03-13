"""
Physics Components — GFN V5
============================
Modular physics components for singularity handling, hysteresis dynamics, and friction.
"""

from gfn.physics.components.singularities import (
    SingularityGate,
    SingularityDetector,
    SingularityRegistry,
)

from gfn.physics.components.hysteresis import (
    HysteresisModule,
    HysteresisState,
    HysteresisRegistry,
)

from gfn.physics.components.friction import (
    FrictionGate,
    AdaptiveFriction,
    FrictionRegistry,
)

from gfn.physics.components.stochasticity import (
    BrownianForce,
    OUDynamicsForce,
)

from gfn.physics.components.curiosity import (
    GeometricCuriosityForce,
)

__all__ = [
    # Singularity components
    'SingularityGate',
    'SingularityDetector', 
    'SingularityRegistry',
    
    # Hysteresis components
    'HysteresisModule',
    'HysteresisState',
    'HysteresisRegistry',
    
    # Friction components
    'FrictionGate',
    'AdaptiveFriction',
    'FrictionRegistry',

    # Stochastic / Exploration components
    'BrownianForce',
    'OUDynamicsForce',
    'GeometricCuriosityForce',
]
