"""
gfn/math/__init__.py
Abstracciones matemáticas para GFN V5.
"""
from gfn.math.distances import geodesic_distance_torus, geodesic_distance_euclidean, wrap_to_pi
from gfn.math.differential import christoffel_contraction, parallel_transport_approx
from gfn.math.physics import ricci_scalar_approx, hamiltonian_energy
from gfn.math.stability import safe_log, safe_norm, entropy

__all__ = [
    # Geometry
    "geodesic_distance_torus", "geodesic_distance_euclidean", "wrap_to_pi",
    # Differential
    "christoffel_contraction", "parallel_transport_approx",
    # Physics
    "ricci_scalar_approx", "hamiltonian_energy",
    # Stability
    "safe_log", "safe_norm", "entropy"
]
