"""Tests for Exponential layer"""

import pytest
import torch
import numpy as np
import numpy.testing
import torch.autograd.gradcheck
from torch.autograd import Variable

from econ_layers.layers import FlexibleSequential

torch.set_printoptions(16)  # to be able to see what is going on


def test_simple_flexible():
    torch.manual_seed(1234)

    mod = FlexibleSequential(2, 3, layers=3, hidden_dim=128, dtype=torch.float64)
    x = torch.tensor([1.0, 1.5], dtype=torch.float64)
    y = mod(x)
    y_out = torch.tensor(
        [0.0573280295009930, -0.0529505785678611, -0.0710645560385077],
        dtype=torch.float64,
    )
    assert torch.all(torch.isclose(y.detach(), y_out))


# Unit testing of the autodiff.  Easy here, but need to be more advanced later
def test_simple_flexible_derivative():
    n_in = 20
    n_out = 3
    mod = FlexibleSequential(n_in, n_out, layers=3, hidden_dim=128, dtype=torch.float64)
    input = (Variable(torch.randn(n_in, dtype=torch.float64), requires_grad=True),)
    test = torch.autograd.gradcheck(mod, input)


## A few other checks on important features.