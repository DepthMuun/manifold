import torch
from gfn.geometry.lowrank import LowRankChristoffel
from gfn.integrators.runge_kutta.euler import EulerIntegrator
from gfn.integrators.runge_kutta.heun import HeunIntegrator
from gfn.integrators.runge_kutta.rk4 import RK4Integrator


def test_integrators_with_constant_force_single_step():
    dim = 4
    physics = {'stability': {'friction': 0.0}}
    christ = LowRankChristoffel(dim, rank=2, physics_config=physics)
    x0 = torch.zeros(1, dim)
    v0 = torch.zeros(1, dim)
    force = torch.ones(1, dim) * 0.25
    dt = 0.1

    euler = EulerIntegrator(christ, dt=dt)
    heun = HeunIntegrator(christ, dt=dt)
    rk4 = RK4Integrator(christ, dt=dt)

    x_e, v_e = euler(x0, v0, force=force, steps=1)
    x_h, v_h = heun(x0, v0, force=force, steps=1)
    x_r, v_r = rk4(x0, v0, force=force, steps=1)

    v_expected = v0 + dt * force
    x_expected_heun_rk4 = x0 + dt * v0 + 0.5 * (dt ** 2) * force
    x_expected_euler = x0 + dt * v0

    assert torch.allclose(v_e, v_expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(v_h, v_expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(v_r, v_expected, atol=1e-6, rtol=1e-6)

    assert torch.allclose(x_e, x_expected_euler, atol=1e-6, rtol=1e-6)
    assert torch.allclose(x_h, x_expected_heun_rk4, atol=1e-6, rtol=1e-6)
    assert torch.allclose(x_r, x_expected_heun_rk4, atol=1e-6, rtol=1e-6)*** End Patch``` } ***!
