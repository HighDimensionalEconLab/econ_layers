#!/usr/bin/env python

"""Tests for Moments layer"""

import pytest
import torch
import numpy
from econ_layers.layers import Moments


def test_moments():
    moments_layer = Moments(5)
    x = torch.tensor([2.0, 1.5])
    x_moments = moments_layer(x.reshape([2, 1]))
    assert torch.all(
        torch.isclose(
            x_moments,
            torch.tensor(
                [[2.0, 4.0, 8.0, 16.0, 32.0], [1.5, 2.25, 3.375, 5.0625, 7.59375]]
            ),
        )
    )
