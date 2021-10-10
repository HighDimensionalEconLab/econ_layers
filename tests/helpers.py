import pytest
import torch
from torch import nn


# helper function to train arbitrary models
@pytest.mark.skip(reason="helper")
def train(model, parameters, target, optimizer, loss_function = nn.MSELoss()):
    optimizer.zero_grad()
    result = model(*parameters)
    loss = loss_function(result, target)
    loss.backward()
    optimizer.step()
    return loss.item()


# helper function to test arbitrary models
@pytest.mark.skip(reason="helper")
def test(model, parameters, target, loss_function = nn.MSELoss()):
    with torch.no_grad():
        result = model(*parameters)
        loss = loss_function(result, target)
        return loss.item()
