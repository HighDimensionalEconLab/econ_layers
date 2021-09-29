import torch
from torch import nn

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
        hidden_bias=True,
        LastActivator=nn.Identity,
        last_bias=True,
        RescalingLayer=None,
        rescaling_layer_kwargs=[]
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
        self.LastActivator = LastActivator
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
            self.Activator(),
            # Add in layers - 1
            *[
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.hidden_bias),
                    self.Activator(),
                )
                for i in range(self.layers - 1)
            ],
            nn.Linear(self.hidden_dim, self.n_out, bias=self.last_bias),
            self.LastActivator()
        )

    def forward(self, input):
        out = self.model(input)  # pass through to the stored net
        if not self.RescalingLayer is None:
            # The rescaling should take n_in -> a n_out x n_out matrix
            # then out = model(input)*rescale(input) 
            return out # TODO
        else:
            return out

    def string(self):
        return self.model.string()  # dispay as
