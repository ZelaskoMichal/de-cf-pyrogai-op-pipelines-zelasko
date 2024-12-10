"""Tests for toolkit."""

import numpy as np
import torch

import template_pipelines.utils.ml_inference.toolkit as toolkit  # noqa I250


def test_auto_encoder():
    """Test AutoEncoder."""
    x = torch.randn((1, 100))

    autoencoder = toolkit.AutoEncoder(n_dim=100)
    encoder_output = autoencoder.encoder(x)
    decoder_output = autoencoder.decoder(encoder_output)
    output = autoencoder(x)

    assert encoder_output.shape == torch.Size([1, 16])
    assert decoder_output.shape == x.shape
    assert output.shape == x.shape
    assert decoder_output.shape == output.shape
    assert isinstance(output, torch.Tensor)
    assert output.dtype == torch.float32


def test_conver_to_tensor():
    """Test convert_to_tensor with a numpy array."""
    array = np.random.randn(1, 100)
    tensor = toolkit.convert_to_tensor(array)

    assert tensor.dtype == torch.float32
    assert tensor.shape == array.shape
