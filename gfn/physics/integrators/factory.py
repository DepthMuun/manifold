"""
IntegratorFactory — GFN V5
Creates integrator instances from PhysicsConfig.
"""

from typing import Optional
from gfn.config.schema import PhysicsConfig
from gfn.interfaces.physics import PhysicsEngine
from gfn.registry import INTEGRATOR_REGISTRY

# Trigger @register_integrator decorators
from gfn.physics.integrators.symplectic import yoshida    # 'yoshida'
from gfn.physics.integrators.symplectic import leapfrog  # 'leapfrog'
from gfn.physics.integrators.symplectic import verlet    # 'verlet'
from gfn.physics.integrators.symplectic import forest_ruth  # 'forest_ruth'
from gfn.physics.integrators.runge_kutta import heun     # 'heun'
from gfn.physics.integrators.runge_kutta import rk4      # 'rk4'


class IntegratorFactory:
    """
    Creates integrators from PhysicsConfig.
    Key: config.stability.integrator_type (e.g., 'yoshida', 'leapfrog', 'heun')
    """

    @staticmethod
    def create(physics_engine: PhysicsEngine, config: Optional[PhysicsConfig] = None):
        config = config or PhysicsConfig()
        integrator_type = getattr(config.stability, 'integrator_type', 'leapfrog').lower()

        available = INTEGRATOR_REGISTRY.list_keys()

        if integrator_type in available:
            cls = INTEGRATOR_REGISTRY.get(integrator_type)
            return cls(physics_engine=physics_engine, config=config)

        print(f"[IntegratorFactory] Warning: '{integrator_type}' not found. Using leapfrog.")
        from gfn.physics.integrators.symplectic.leapfrog import LeapfrogIntegrator
        return LeapfrogIntegrator(physics_engine=physics_engine, config=config)
