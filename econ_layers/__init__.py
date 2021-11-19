"""Top-level package for econ_layers."""
import warnings

__author__ = """Various Contributors"""
__email__ = "jesseperla@gmail.com"
__version__ = "0.0.1"


warnings.filterwarnings(
    "ignore",
    ".*does not have many workers which may be.*",
)
warnings.filterwarnings(
    "ignore",
    ".*is smaller than the logging.*",
)
warnings.filterwarnings(
    "ignore",
    ".*GPU available but not used.*",
)