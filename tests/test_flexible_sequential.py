"""Tests for Exponential layer"""

import pytest
import torch
import numpy as np
import numpy.testing
import torch.autograd.gradcheck
from torch.autograd import Variable

from econ_layers.layers import FlexibleSequential

torch.set_printoptions(16)  # to be able to see what is going on


# Unit testing of the autodiff.  Easy here, but need to be more advanced later
def test_simple_flexible_derivative():
    n_in = 20
    n_out = 3
    mod = FlexibleSequential(n_in, n_out, layers=3, hidden_dim=128).double()
    input = (Variable(torch.randn(n_in).double(), requires_grad=True),)
    test = torch.autograd.gradcheck(mod, input)


## A few other checks on important features.