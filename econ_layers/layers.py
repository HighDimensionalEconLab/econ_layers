import torch
from torch import nn


# rescaling by a specific element of a given input
class InputRescaling(nn.Module):
    def __init__(self, rescale_index):
        super().__init__()
        self.rescale_index = rescale_index
        
    def forward(self, x, y):
        return y / x[self.rescale_index]
    

# Scalar rescaling.  Only one parameter.
class ScalarExponentialRescaling(nn.Module):
    def __init__(self, n_in = 1):
        super().__init__()
        self.n_in = n_in
        self.weight = torch.nn.Parameter(torch.Tensor(1)) # only one parameter to "learn"
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
        n_in,
        n_out,
        layers,
        hidden_dim,
        Activator=nn.ReLU,
        activator_kwargs={},
        hidden_bias=True,
        LastActivator=nn.Identity,
        last_activator_kwargs={},
        last_bias=True,
        RescalingLayer=None,
        rescaling_layer_kwargs={}
    ):
        """
        Init method.
        """
        super().__init__()  # init the base class
        self.n_in = n_in
        self.n_out = n_out
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.Activator = Activator
        self.activator_kwargs = activator_kwargs
        self.LastActivator = LastActivator
        self.last_activator_kwargs = last_activator_kwargs
        self.hidden_bias = hidden_bias
        self.last_bias = last_bias
        self.rescaling_layer_args = rescaling_layer_kwargs
        self.RescalingLayer = RescalingLayer # must map n_in to (n_out x n_out) matrix

        if not self.RescalingLayer == None:
            self.rescale = RescalingLayer(**rescaling_layer_kwargs)  # construct as required
        else:
            self.rescale = None

        # Constructor
        self.model = nn.Sequential(
            nn.Linear(self.n_in, self.hidden_dim, bias=self.hidden_bias),
            self.Activator(**self.activator_kwargs),
            # Add in layers - 1
            *[
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.hidden_bias),
                    self.Activator(**self.activator_kwargs),
                )
                for i in range(self.layers - 1)
            ],
            nn.Linear(self.hidden_dim, self.n_out, bias=self.last_bias),
            self.LastActivator(**self.last_activator_kwargs)
        )

    def forward(self, input):
        out = self.model(input)  # pass through to the stored net
        if not self.RescalingLayer is None:
            # The rescaling should take n_in -> a n_out x n_out matrix
            # then out = model(input)*rescale(input) 
            return self.rescale(input, out)
        else:
            return out

    def string(self):
        return self.model.string()  # dispay as
