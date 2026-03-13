"""
gfn/geometry/__init__.py
Public API for the geometry module — GFN V5
"""

# Base and factory
from gfn.geometry.base import BaseGeometry
from gfn.geometry.factory import GeometryFactory

# Concrete geometries (imports trigger @register_geometry decorators)
from gfn.geometry.euclidean import EuclideanGeometry
from gfn.geometry.torus import ToroidalRiemannianGeometry, FlatToroidalRiemannianGeometry
from gfn.geometry.low_rank import LowRankRiemannianGeometry, PaperLowRankRiemannianGeometry
from gfn.geometry.adaptive import AdaptiveRiemannianGeometry
from gfn.geometry.reactive import ReactiveRiemannianGeometry
from gfn.geometry.hyperbolic import HyperRiemannianGeometry
from gfn.geometry.holographic import HolographicRiemannianGeometry
from gfn.geometry.spherical import SphericalGeometry
from gfn.geometry.hierarchical import HierarchicalGeometry

# Re-export FrictionGate from unified physics.components location
from gfn.physics.components.friction import FrictionGate

__all__ = [
    # Base
    "BaseGeometry",
    "GeometryFactory",
    # Implementations
    "EuclideanGeometry",
    "ToroidalRiemannianGeometry",
    "FlatToroidalRiemannianGeometry",
    "LowRankRiemannianGeometry",
    "PaperLowRankRiemannianGeometry",
    "AdaptiveRiemannianGeometry",
    "ReactiveRiemannianGeometry",
    "HyperRiemannianGeometry",
    "HolographicRiemannianGeometry",
    "SphericalGeometry",
    "HierarchicalGeometry",
    # Shared components
    "FrictionGate",
]
