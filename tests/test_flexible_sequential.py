"""Tests for Exponential layer"""

import pytest
import torch
import numpy as np
import torch
from torch import nn
import numpy.testing
import torch.autograd.gradcheck
from torch.autograd import Variable

from econ_layers.layers import (
    FlexibleSequential,
    RescaleOutputsByInput,
    ScalarExponentialRescaling,
)

torch.set_printoptions(16)  # to be able to see what is going on


# Unit testing of the autodiff.  Easy here, but need to be more advanced later
def test_simple_flexible_derivative():
    n_in = 20
    n_out = 3
    mod = FlexibleSequential(n_in, n_out, layers=3, hidden_dim=128).double()
    input = (Variable(torch.randn(n_in).double(), requires_grad=True),)
    assert torch.autograd.gradcheck(mod, input)


## A few other checks on important features.

# Unit testing of the autodiff.  Easy here, but need to be more advanced later
def test_simple_flexible_derivative_softplus():
    n_in = 20
    n_out = 3
    mod = FlexibleSequential(
        n_in,
        n_out,
        layers=3,
        hidden_dim=128,
        last_activator=nn.Softplus(beta=2.0),
    ).double()
    input = (Variable(torch.randn(n_in).double(), requires_grad=True),)
    assert torch.autograd.gradcheck(mod, input)


# Unit test with an integrated scalar rescale layer in the FlexibleSequential
def test_simple_flexible_derivative_rescale():
    n_in = 1
    n_out = 2
    mod = FlexibleSequential(
        n_in,
        n_out,
        layers=3,
        hidden_dim=128,
        OutputRescalingLayer=ScalarExponentialRescaling(),
    ).double()
    input = (Variable(torch.randn(n_in).double(), requires_grad=True),)
    assert torch.autograd.gradcheck(mod, input)


# Unit test with rescaling by one of the inputs
def test_simple_flexible_derivative_input_rescale():
    n_in = 2
    n_out = 1
    mod = FlexibleSequential(
        n_in,
        n_out,
        layers=3,
        hidden_dim=128,
        OutputRescalingLayer=RescaleOutputsByInput(rescale_index=0),
    ).double()
    input = (Variable(torch.randn(n_in).double(), requires_grad=True),)
    assert torch.autograd.gradcheck(mod, input)
