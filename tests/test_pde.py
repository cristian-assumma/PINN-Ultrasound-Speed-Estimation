import torch
import math
import pytest

def diff(u, x, order=1):
    """
    Standalone version of the Autograd differentiation mechanism used in loss.py.
    Extracts the analytical derivative of the network's computational graph.
    """
    grads = torch.autograd.grad(u, x,
                                grad_outputs=torch.ones_like(u),
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True)[0]
    if order == 1:
        return grads
    grads_2 = torch.autograd.grad(grads, x,
                                  grad_outputs=torch.ones_like(grads),
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
    return grads_2

def test_autograd_first_derivative():
    """Verify that PyTorch Autograd correctly computes d/dx sin(x) = cos(x)"""
    x = torch.linspace(-math.pi, math.pi, 100, requires_grad=True)
    u = torch.sin(x)

    u_x = diff(u, x, order=1)
    expected = torch.cos(x)

    # Assert maximum absolute error is practically zero
    assert torch.allclose(u_x, expected, atol=1e-5)

def test_autograd_second_derivative():
    """Verify that PyTorch Autograd correctly computes d^2/dx^2 sin(x) = -sin(x)"""
    x = torch.linspace(-math.pi, math.pi, 100, requires_grad=True)
    u = torch.sin(x)

    u_xx = diff(u, x, order=2)
    expected = -torch.sin(x)

    assert torch.allclose(u_xx, expected, atol=1e-5)


def test_acoustic_wave_equation_residual():
    """
    Core physical validation: Test the PDE residual for a 1D acoustic wave.
    Uses Float64 to prevent binary quantization errors when squaring c=1540.0.
    """
    # Cast to float64 to handle large physical constants without precision loss
    x = torch.rand(100, dtype=torch.float64, requires_grad=True)
    t = torch.rand(100, dtype=torch.float64, requires_grad=True)

    # Speed of sound in tissue (m/s)
    c = 1540.0

    # Forward analytical wave
    u = torch.sin(x - c * t)

    # Compute 2nd order derivatives via computational graph
    u_tt = diff(u, t, order=2)
    u_xx = diff(u, x, order=2)

    # Inverse scattering PDE residual
    pde_residual = u_tt - (c ** 2) * u_xx

    # The residual must be mathematically zero
    assert torch.allclose(pde_residual, torch.zeros_like(pde_residual), atol=1e-5)