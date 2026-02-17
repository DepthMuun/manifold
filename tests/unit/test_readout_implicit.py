import torch
from gfn.readouts.implicit import ImplicitReadout


def test_readout_shapes_flat_and_torus():
    B, L, D = 2, 3, 4
    coord = 6
    x = torch.randn(B, L, D)

    r0 = ImplicitReadout(D, coord, topology=0)
    r1 = ImplicitReadout(D, coord, topology=1)

    y0 = r0(x)
    y1 = r1(x)

    assert y0.shape == (B, L, coord)
    assert y1.shape == (B, L, coord)
