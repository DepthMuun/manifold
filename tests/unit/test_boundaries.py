import math
import torch
from gfn.geometry.boundaries import apply_boundary_python


def test_apply_boundary_torus_range_and_invariance():
    pi = math.pi
    two_pi = 2 * pi
    x = torch.tensor([[-4 * two_pi, -pi / 2, 0.0, pi, 3 * pi]])
    y = apply_boundary_python(x, topology_id=1)
    assert torch.all(y >= 0.0)
    assert torch.all(y < two_pi + 1e-6)

    k = 5
    x_shift = x + k * two_pi
    y_shift = apply_boundary_python(x_shift, topology_id=1)
    assert torch.allclose(y, y_shift, atol=1e-6, rtol=1e-6)


def test_apply_boundary_euclidean_no_change():
    x = torch.randn(2, 4)
    y = apply_boundary_python(x, topology_id=0)
    assert torch.allclose(x, y)***
