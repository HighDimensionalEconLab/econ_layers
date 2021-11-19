import torch
from copy import deepcopy
from torch import nn
from typing import Optional

# rescaling by a specific element of a given input
class RescaleOutputsByInput(nn.Module):
    def __init__(self, rescale_index: int):
        super().__init__()
        self.rescale_index = rescale_index

    def forward(self, x, y):
        if x.dim() == 1:
            return x[self.rescale_index] * y
        else:
            return x[:, [self.rescale_index]] * y


# [[z_0,k_0], [z]]


# Scalar rescaling.  Only one parameter.
class ScalarExponentialRescaling(nn.Module):
    def __init__(self, n_in: int = 1):
        super().__init__()
        self.n_in = n_in
        self.weight = torch.nn.Parameter(
            torch.Tensor(1)
        )  # only one parameter to "learn"
        self.reset_parameters()

    def reset_parameters(self):
        # Lets start at zero, but later could have option
        torch.nn.init.zeros_(self.weight)

    def forward(self, x, y):
        exp_x = torch.exp(self.weight * x)  # exponential of input
        return torch.mul(exp_x, y)


# There is no exponential layer in pytorch, so this adds one.
class Exponential(nn.Module):
    def __init__(self):
        """
        Init method.
        """
        super().__init__()  # init the base class

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return torch.exp(input)


class FlexibleSequential(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        layers: int,
        hidden_dim: int,
        activator: Optional[nn.Module] = nn.ReLU(),
        hidden_bias: bool = True,
        last_activator: Optional[nn.Module] = nn.Identity(),
        last_bias=True,
        rescaling_layer: Optional[nn.Module] = None,
    ):
        """
        Init method.
        """
        super().__init__()  # init the base class
        self.n_in = n_in
        self.n_out = n_out
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.activator = activator
        self.last_activator = last_activator
        self.hidden_bias = hidden_bias
        self.last_bias = last_bias
        self.rescaling_layer = rescaling_layer

        # Constructor
        self.model = nn.Sequential(
            nn.Linear(self.n_in, self.hidden_dim, bias=self.hidden_bias),
            self.activator,
            # Add in layers - 1
            *[
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.hidden_bias),
                    deepcopy(
                        self.activator
                    ),  # use as prototype. Cannot support learnable paramters in activator
                )
                for i in range(self.layers - 1)
            ],
            nn.Linear(self.hidden_dim, self.n_out, bias=self.last_bias),
            self.last_activator
        )

    def forward(self, input):
        out = self.model(input)  # pass through to the stored net
        if not self.rescaling_layer is None:
            # The rescaling should take n_in -> a n_out x n_out matrix
            # then out = model(input)*rescale(input)
            return self.rescaling_layer(input, out)
        else:
            return out

    def string(self):
        return self.model.string()  # dispay as
