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
        LastLayer=nn.Identity,
        hidden_bias=True,
        last_bias=True,
        device=None,
        dtype=None,
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
        self.LastLayer = LastLayer
        self.hidden_bias = hidden_bias
        self.last_bias = last_bias
        self.device = device
        self.dtype = dtype

        # Constructor
        self.model = nn.Sequential(
            nn.Linear(self.n_in, self.hidden_dim, bias=self.hidden_bias, device = self.device, dtype = self.dtype),
            self.Activator(),
            # Add in layers - 1
            *[
                nn.Sequential(
                    nn.Linear(
                        self.hidden_dim,
                        self.hidden_dim,
                        bias=self.hidden_bias,
                        device = self.device, dtype = self.dtype
                    ),
                    self.Activator(),
                )
                for i in range(self.layers - 1)
            ],
            nn.Linear(self.hidden_dim, self.n_out, bias=self.last_bias, device = self.device, dtype = self.dtype),
            self.LastLayer()
        )

    def forward(self, input):
        return self.model(input)  # pass through to the stored net

    def string(self):
        return self.model.string()  # dispay as
