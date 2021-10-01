# Pytorch Layers for Economics Applications

[![image](https://img.shields.io/pypi/v/econ_layers.svg)](https://pypi.python.org/pypi/econ_layers) [![Build Status](https://github.com/HighDimensionalEconLab/econ_layers/workflows/build/badge.svg)](https://github.com/HighDimensionalEconLab/econ_layers/actions)

## Pytorch 


-   Documentation: https://HighDimensionalEconLab.github.io/econ_layers

## Features

- Exponential layer
- Flexible multi-layer neural network with optional nonlinear last layer


## Development

To publish a new relase to pypi,
1. Ensure that the CI is passing
2. Modify [setup.py](setup.py#L56) to increment the minor version number, or a major version number once API stability is enforced
3. Choose the "Releases" on the github page, then "Draft a new relase"
4. Click on "Choose a tag" and then type a new release tag with the `v` followed by the version number you modified to be consistent
5. After you choose "Publish Release" it will automatically push to pypi, and you can change compatability bounds in downstream packages as required.

## Credits

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template.
