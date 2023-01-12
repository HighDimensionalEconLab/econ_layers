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
    DeepSet,
    DeepSetMoments,
)

torch.set_printoptions(16)  # to be able to see what is going on


# Unit testing of the autodiff.  Easy here, but need to be more advanced later
def test_deep_set_derivative():
    n_in = 1 # one dimensional state per "agent".  Only n_in = 1 is supported right now
    N = 5 # number of "agents"
    n_out = 2
    mod = DeepSet(n_in, n_out, L = 2, phi_layers=2, phi_hidden_dim=32, rho_layers=2, rho_hidden_dim=32).double()
    batches = 10
    input = (Variable(torch.randn(N, batches).double(), requires_grad=True),)
    assert torch.autograd.gradcheck(mod, input)


def test_deep_set_moments_derivative():
    n_in = 1 # one dimensional state per "agent".  Only n_in = 1 is supported right now
    N = 5 # number of "agents"
    n_out = 2
    num_moments = 3
    mod = DeepSetMoments(n_in, n_out, L = num_moments, rho_layers=2, rho_hidden_dim=32).double()
    batches = 10
    input = (Variable(torch.randn(N, batches).double(), requires_grad=True),)
    assert torch.autograd.gradcheck(mod, input)    