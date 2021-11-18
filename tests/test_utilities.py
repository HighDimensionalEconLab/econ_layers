"""Tests for utility functions"""

import pytest
import torch
import numpy as np
from econ_layers.utilities import squeeze_cpu, dict_to_cpu

# test squeeze_cpu

def test_squeeze_cpu_sanity():
    x = squeeze_cpu(torch.tensor([1.2, 3.4]))
    assert(np.allclose(x, np.array([1.2, 3.4])))

def test_squeeze_cpu_squeezed():
    x = squeeze_cpu(torch.tensor([[[1.2, 3.4]], [[5.6, 7.8]]]))
    assert(np.allclose(x, np.array([[1.2, 3.4], [5.6, 7.8]])))

def test_squeeze_cpu_non_tensor():
    x = squeeze_cpu(np.array([[[1.2, 3.4]], [[5.6, 7.8]]]))
    assert(np.allclose(x, np.array([[[1.2, 3.4]], [[5.6, 7.8]]])))

# test dict_to_cpu

def test_dict_to_cpu():
    d = {
        "simple_tensor": torch.tensor([7.1, 4.2]),
        "tensor_unsqueezed": torch.tensor([[[2.1, 4.3]], [[6.5, 8.7]]]),
        "numpy_unsqueezed": np.array([[[2.1, 3.4]], [[5.6, 8.7]]]),
        "a_float": 4.2
    }
    expected_dict = {
        "simple_tensor": [7.1, 4.2],
        "tensor_unsqueezed": [[2.1, 4.3], [6.5, 8.7]],
        "numpy_unsqueezed": [[[2.1, 3.4]], [[5.6, 8.7]]],
        "a_float": 4.2
    }
    assert(d[key] == expected_dict[key] for key in d)
