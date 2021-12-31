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

    
def test_moments_broadcast():
    num_moments = 5
    test_data = torch.tensor([[1.0, 2, 3, 4], [5.0, 6, 7, 8]])
    num_batches, N = test_data.shape
    moments_layer = Moments(num_moments)
    generated_data = torch.stack(
        [
            torch.mean(moments_layer(test_data[i, :].reshape([N, 1])), 0)
            for i in range(num_batches)
        ]
    )
    expected_data = torch.stack(
        [
            torch.mean(
                torch.stack(
                    [
                        torch.tensor(
                            [elt ** moment for moment in range(1, num_moments + 1)]
                        )
                        for elt in test_data[i, :]
                    ]
                ),
                0,
            )
            for i in range(num_batches)
        ]
    )

    assert torch.all(
        torch.isclose(
            generated_data,
            expected_data,
        )
    )
    
    assert torch.all(
        torch.isclose(
            generated_data,
            torch.tensor(
                [
                    [2.5e00, 7.5e00, 2.5e01, 8.85e01, 3.25e02],
                    [6.5e00, 4.35e01, 2.99e02, 2.1045e03, 1.5119e04],
                ]
            ),
        )
    )
