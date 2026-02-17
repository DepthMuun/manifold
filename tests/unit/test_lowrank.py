import torch
from gfn.geometry.lowrank import LowRankChristoffel


def test_lowrank_shapes_and_friction_velocity_dependence():
    dim = 4
    physics = {'stability': {'friction': 1.0, 'velocity_friction_scale': 1.0}}
    christ = LowRankChristoffel(dim, rank=2, physics_config=physics)
    christ.return_friction_separately = True

    x = torch.zeros(1, dim)
    v_small = torch.ones(1, dim) * 0.1
    v_big = torch.ones(1, dim) * 10.0

    gamma_s, mu_s = christ(v_small, x)
    gamma_b, mu_b = christ(v_big, x)

    assert gamma_s.shape == (1, dim)
    assert mu_s.shape == (1, dim)

    assert mu_b.mean() > mu_s.mean() - 1e-6
