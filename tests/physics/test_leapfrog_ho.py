import torch
from gfn.geometry.lowrank import LowRankChristoffel
from gfn.integrators.symplectic.leapfrog import LeapfrogIntegrator


def test_leapfrog_energy_conservation_simple_ho():
    dim = 2
    physics = {'stability': {'friction': 0.0}}
    christ = LowRankChristoffel(dim, rank=1, physics_config=physics)
    dt = 0.01
    integrator = LeapfrogIntegrator(christ, dt=dt)

    x = torch.tensor([[1.0, 0.0]])
    v = torch.tensor([[0.0, 1.0]])

    def energy(x, v):
        return 0.5 * (x.pow(2).sum(dim=-1) + v.pow(2).sum(dim=-1))

    e0 = energy(x, v)

    steps = 200
    for _ in range(steps):
        force = -x
        x, v = integrator(x, v, force=force, steps=1)

    e1 = energy(x, v)
    diff = torch.abs(e1 - e0).item()
    assert diff < 1e-2
