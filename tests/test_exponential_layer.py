#!/usr/bin/env python

"""Tests for Exponential layer"""

import pytest
import torch
import numpy
from econ_layers.layers import Exponential


def test_exponential():
    exp_layer = Exponential()
    x = torch.tensor([1.0, 1.5])
    exp_x = exp_layer(x)
    assert(torch.all(torch.isclose(exp_x,  torch.tensor([2.7183, 4.4817]))))