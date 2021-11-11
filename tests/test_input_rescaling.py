#!/usr/bin/env python

"""Tests for InputRescaling layer"""

import pytest
import torch
import numpy
from econ_layers.layers import RescaleOutputsByInput


def test_input_rescaling():

    x = torch.cartesian_prod(
        torch.linspace(
            0.0,
            10.0,
            steps=20,
        ),
        torch.linspace(
            1.0,
            3.0,
            steps=5,
        ),
    )

    input_layer_0 = RescaleOutputsByInput(0)
    input_layer_1 = RescaleOutputsByInput(1)
    y_1 = x[:, [0]]
    y_0 = x[:, [1]]
    input_mult_0 = input_layer_0(x, y_0)
    input_mult_1 = input_layer_1(x, y_1)

    assert torch.all(torch.isclose(input_mult_0, x[:, [0]] * x[:, [1]]))
    assert torch.all(torch.isclose(input_mult_1, x[:, [0]] * x[:, [1]]))


