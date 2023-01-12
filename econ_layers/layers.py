import torch
from copy import deepcopy
from torch import nn
from typing import Optional
from jsonargparse import lazy_instance


# produces the first m moments of a given input
class Moments(nn.Module):
    def __init__(
        self,
        n_moments: int,
    ):
        super().__init__()
        self.n_moments = n_moments

    def forward(self, input):
        return torch.cat([input.pow(m) for m in torch.arange(1, self.n_moments + 1)], 1)


# rescaling by a specific element of a given input
class RescaleOutputsByInput(nn.Module):
    def __init__(self, rescale_index: int = 0, bias=False):
        super().__init__()
        self.rescale_index = rescale_index
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(1))  # only a scalar here
            torch.nn.init.ones_(self.bias)
        else:
            self.bias = 0.0  # register_parameter('bias', None) # necessary?

    def forward(self, x, y):
        if x.dim() == 1:
            return x[self.rescale_index] * y + self.bias
        else:
            return x[:, [self.rescale_index]] * y + self.bias


# assuming 2D data
class RescaleAllInputsbyInput(nn.Module):
    def __init__(self, rescale_index: int = 0):
        super().__init__()
        self.rescale_index = rescale_index

    def forward(self, x):

        if x.dim() == 1:
            size = x.size()[0]
            rescale_scalar = 1 / x[self.rescale_index]
            rescal_matrix = rescale_scalar.repeat(1, size)
            rescal_matrix[self.rescale_index] = 1.0
            return x * rescal_matrix
        else:
            size = x.size()[1]
            rescale_scalar = 1 / x[:, [self.rescale_index]]
            rescal_matrix = rescale_scalar.repeat(1, size)
            rescal_matrix[:, [self.rescale_index]] = 1.0

            return x * rescal_matrix


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
        hidden_dim: int = 128,
        activator: Optional[nn.Module] = lazy_instance(nn.ReLU),
        hidden_bias: bool = True,
        last_activator: Optional[nn.Module] = lazy_instance(nn.Identity),
        last_bias=True,
        OutputRescalingLayer: Optional[nn.Module] = None,
        InputRescalingLayer: Optional[nn.Module] = None,
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
        self.OutputRescalingLayer = OutputRescalingLayer
        self.InputRescalingLayer = InputRescalingLayer
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
        if not self.InputRescalingLayer is None:
            input = self.InputRescalingLayer(input)

        out = self.model(input)  # pass through to the stored net
        if not self.OutputRescalingLayer is None:
            # The rescaling should take n_in -> a n_out x n_out matrix
            # then out = model(input)*rescale(input)
            return self.OutputRescalingLayer(input, out)
        else:
            return out

class DeepSet(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        L: int,        
        phi_layers: int,
        rho_layers: int,
        phi_hidden_dim: int = 128,
        rho_hidden_dim: int = 128,
        phi_activator: Optional[nn.Module] = lazy_instance(nn.ReLU),
        phi_hidden_bias: bool = True,
        phi_last_activator: Optional[nn.Module] = lazy_instance(nn.Identity),
        phi_last_bias=True,
        rho_activator: Optional[nn.Module] = lazy_instance(nn.ReLU),
        rho_hidden_bias: bool = True,
        rho_last_activator: Optional[nn.Module] = lazy_instance(nn.Identity),
        rho_last_bias=True,        
        OutputRescalingLayer: Optional[nn.Module] = None,
        InputRescalingLayer: Optional[nn.Module] = None,
    ):
        """
        Init method.
        """
        assert n_in == 1  # only supporting univariate states for now
        super().__init__()  # init the base class
        self.rho = FlexibleSequential(
            L,
            n_out,
            rho_layers,
            rho_hidden_dim,
            rho_activator,
            rho_hidden_bias,
            rho_last_activator,
            rho_last_bias,
            OutputRescalingLayer = OutputRescalingLayer,
        )

        self.phi = FlexibleSequential(
            n_in,
            L,
            phi_layers,
            phi_hidden_dim,
            phi_activator,
            phi_hidden_bias,
            phi_last_activator,
            phi_last_bias,
            InputRescalingLayer = InputRescalingLayer,
        )

    def forward(self, X):
        num_batches, N = X.shape
        phi_X = torch.stack(
            [torch.mean(self.phi(X[i, :].reshape([N, 1])), 0) for i in range(num_batches)]
        )
        return self.rho(phi_X)



class DeepSetMoments(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        L: int,        
        rho_layers: int,
        rho_hidden_dim: int = 128,
        rho_activator: Optional[nn.Module] = lazy_instance(nn.ReLU),
        rho_hidden_bias: bool = True,
        rho_last_activator: Optional[nn.Module] = lazy_instance(nn.Identity),
        rho_last_bias=True,        
        OutputRescalingLayer: Optional[nn.Module] = None,
    ):
        """
        Init method.
        """
        assert n_in == 1  # only supporting univariate states for now
        super().__init__()  # init the base class
        self.rho = FlexibleSequential(
            L,
            n_out,
            rho_layers,
            rho_hidden_dim,
            rho_activator,
            rho_hidden_bias,
            rho_last_activator,
            rho_last_bias,
            OutputRescalingLayer = OutputRescalingLayer,
        )

        self.phi = Moments(L)

    def forward(self, X):
        num_batches, N = X.shape
        phi_X = torch.stack(
            [torch.mean(self.phi(X[i, :].reshape([N, 1])), 0) for i in range(num_batches)]
        )
        return self.rho(phi_X)
