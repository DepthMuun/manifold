import torch
from gfn.losses.hamiltonian import hamiltonian_loss


def test_hamiltonian_loss_none_mode_zero():
    v = [torch.zeros(2, 4), torch.zeros(2, 4)]
    loss = hamiltonian_loss(v, mode='none')
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-8)


def test_hamiltonian_loss_relative_zero_when_equal_energies():
    v = [torch.ones(2, 3), torch.ones(2, 3)]
    loss = hamiltonian_loss(v, mode='relative')
    assert torch.isfinite(loss)
    assert loss < 1e-6
