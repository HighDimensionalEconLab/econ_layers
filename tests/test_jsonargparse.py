import pytest
import sys
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI


class TestModel(pl.LightningModule):
    def __init__(self, ml_model: nn.Module):
        pass


def test_edit_layers():
    sys.argv = [
        "cli.py",
        "--trainer.max_epochs=5",
        "--model.ml_model.init_args.layers=4",
    ]
    cli = LightningCLI(
        TestModel,
        run=False,
        parser_kwargs={
            "default_config_files": ["tests/default_jsonargparse_test_1.yaml"]
        },
    )
    ml_model_expected = {
        "class_path": "econ_layers.layers.FlexibleSequential",
        "init_args": {
            "n_in": 1,
            "n_out": 1,
            "layers": 4,
            "hidden_dim": 122,
            "activator": {
                "class_path": "torch.nn.modules.activation.ReLU",
                "init_args": {"inplace": False},
            },
            "hidden_bias": True,
            "last_activator": {"class_path": "torch.nn.modules.linear.Identity"},
            "last_bias": True,
            "rescaling_layer": None,
        },
    }
    assert cli.config["model"]["ml_model"] == ml_model_expected
