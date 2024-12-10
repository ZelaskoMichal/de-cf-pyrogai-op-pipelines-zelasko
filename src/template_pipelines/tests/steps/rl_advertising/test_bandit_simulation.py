"""Tests for bandit simulation."""

import logging
import os
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
from torch import nn

from template_pipelines.steps.rl_advertising.bandit_simulation import BanditSimulation


@pytest.fixture(scope="function")
def fixture_bandit_simulation():
    """Fixture for the Bandit Simulation step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        bs = BanditSimulation()
        bs.mlflow = MagicMock()
        bs.outputs = {}
        yield bs


class MockModel(nn.Module):
    """A mock neural network model for testing purposes."""

    def __init__(self):
        """Initialize the MockModel instance."""
        super().__init__()

    def forward(self, x):
        """Perform a forward pass through the network."""
        return torch.tensor([1.0])


def mock_generator():
    """A mock data generator."""
    while True:
        yield {"feat1": 1, "feat2": 1, "education": "graduate"}, [1, 1]


@patch("template_pipelines.steps.rl_advertising.bandit_simulation.update_model")
@patch("template_pipelines.steps.rl_advertising.bandit_simulation.simulate_ad_inventory")
def test_run_simulation(mock_simulate_ad_inventory, mock_update_model, fixture_bandit_simulation):
    """Test run_simulation."""
    model = MockModel()
    optimizer = None
    loss = None
    generator = mock_generator()
    ad_click_probs = {"graduate": {"graduate": 0.8}}
    simulation_size = 2
    model_update_freq = 1

    mock_simulate_ad_inventory.return_value = ["graduate"]
    fixture_bandit_simulation.logger = logging

    res = fixture_bandit_simulation.run_simulation(
        model, optimizer, loss, generator, ad_click_probs, simulation_size, model_update_freq
    )

    assert mock_simulate_ad_inventory.call_count == simulation_size
    assert mock_update_model.call_count == simulation_size / model_update_freq
    assert isinstance(res, list)
    for element in res:
        assert isinstance(element, float)


@patch("template_pipelines.steps.rl_advertising.bandit_simulation.torch.save")
def test_save_model(mock_torch_save, fixture_bandit_simulation):
    """Test save_model."""
    model = MockModel()
    model_name = "mock_model"
    output_dir = "test"
    mock_torch_save.return_value = True
    fixture_bandit_simulation.logger = logging

    with TestCase().assertLogs(level="INFO") as log:
        fixture_bandit_simulation.save_model(model, model_name, output_dir)
    msg = log.records[-1].msg
    model_path = os.path.join(output_dir, f"{model_name}.pth")

    fixture_bandit_simulation.mlflow.pytorch.log_model.assert_called_once()
    assert mock_torch_save.asser_called_with(model.state_dict(), model_path)
    assert "The model has been trained and saved to:" in msg


@patch.object(BanditSimulation, "save_model")
@patch("template_pipelines.steps.rl_advertising.bandit_simulation.visualize_line_plot")
@patch("template_pipelines.steps.rl_advertising.bandit_simulation.os.makedirs")
@patch("template_pipelines.steps.rl_advertising.bandit_simulation.update_model")
@patch("template_pipelines.steps.rl_advertising.bandit_simulation.simulate_ad_inventory")
@patch("template_pipelines.steps.rl_advertising.bandit_simulation.build_model")
@patch("template_pipelines.steps.rl_advertising.bandit_simulation.pd.read_parquet")
def test_bandit_simulation_run(
    mock_pd_read_parquet,
    mock_build_model,
    mock_simulate_ad_inventory,
    mock_update_model,
    mock_os_makedirs,
    mock_visualize_line_plot,
    mock_save_model,
    fixture_bandit_simulation,
):
    """Test the Bandit Simulation run method."""
    data_dict, _ = next(mock_generator())
    mock_pd_read_parquet.return_value = pd.DataFrame([data_dict])
    mock_build_model.return_value = (MockModel(), None, None)
    mock_simulate_ad_inventory.return_value = ["graduate"]

    fixture_bandit_simulation.config = {
        "rl_advertising": {"dropout_levels": [0.1], "simulation_size": 2, "model_update_freq": 1}
    }
    fixture_bandit_simulation.run()

    mock_pd_read_parquet.assert_called_once()
    mock_build_model.assert_called()
    mock_os_makedirs.assert_called_once()
    mock_visualize_line_plot.assert_called_once()
    fixture_bandit_simulation.mlflow.log_artifact.assert_called_once()
    mock_save_model.assert_called_once()
    fixture_bandit_simulation.logger.info.assert_called()
