"""Tests for Exponential Rescaling layer"""

import pytest
import numpy as np
import torch
from torch import nn

from econ_layers.layers import DiagonalExponentialRescaling, ScalarExponentialRescaling
from tests.helpers import train, test

torch.set_printoptions(16)  # to be able to see what is going on
torch.manual_seed(0)
learning_rate = 1.0e-2
test_loss_bound = 1.0e-15


"""
---------------------------------------------------------------------
------------ Tests for ScalarExponentialRescaling model -------------
---------------------------------------------------------------------
"""


# test that scalar model successfully retrieves the model parameter
def test_exponential_scalar():
    num_epochs = 1000
    n_in = 100
    parameter = 0.02

    x = torch.linspace(0, 10, steps = n_in).double()
    exp_x = torch.exp(parameter * x)
    y = 10 * torch.rand(n_in)
    target = torch.mul(exp_x, y)
    model = ScalarExponentialRescaling().double()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        train(model, (x, y), target, optimizer)

    assert(model.weights.item() == parameter)
