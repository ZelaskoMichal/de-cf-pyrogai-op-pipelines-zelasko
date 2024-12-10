"""Tests for bandit evaluation."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
from torch import nn

from template_pipelines.steps.rl_advertising.bandit_evaluation import BanditEvaluation


@pytest.fixture(scope="function")
def fixture_bandit_evaluation():
    """Fixture for the Bandit Evaluation step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        be = BanditEvaluation()
        yield be


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


@pytest.mark.parametrize(
    "number, multiple, expected, raise_division_by_zero_error",
    [(9999, 100, 9900, False), (105, 10, 100, False), (10, 0, None, True)],
)
def test_round_nearest_multiple(
    number, multiple, expected, raise_division_by_zero_error, fixture_bandit_evaluation
):
    """Test for round_nearest_multiple."""
    if raise_division_by_zero_error:
        with pytest.raises(ZeroDivisionError):
            fixture_bandit_evaluation.round_nearest_multiple(number, multiple)
    else:
        res = fixture_bandit_evaluation.round_nearest_multiple(number, multiple)
        assert isinstance(res, int)
        assert res == expected


@patch("template_pipelines.steps.rl_advertising.bandit_evaluation.simulate_ad_inventory")
def test_run_evaluation(mock_simulate_ad_inventory, fixture_bandit_evaluation):
    """Test for run_evaluation."""
    model = MockModel()
    generator = mock_generator()
    ad_click_probs = {"graduate": {"graduate": 0.8}}
    test_size = 1
    mock_simulate_ad_inventory.return_value = ["graduate"]

    bandit_rewards, random_rewards = fixture_bandit_evaluation.run_evaluation(
        model, generator, ad_click_probs, test_size
    )

    assert isinstance(bandit_rewards, list)
    assert isinstance(random_rewards, list)
    assert len(bandit_rewards) > 0
    assert len(bandit_rewards) == len(random_rewards)
    assert int(bandit_rewards[0]) == 0 or int(bandit_rewards[0]) == 1
    assert int(random_rewards[0]) == 0 or int(random_rewards[0]) == 1


@patch("template_pipelines.steps.rl_advertising.bandit_evaluation.visualize_line_plot")
@patch("template_pipelines.steps.rl_advertising.bandit_evaluation.os.makedirs")
@patch.object(BanditEvaluation, "round_nearest_multiple")
@patch("template_pipelines.steps.rl_advertising.bandit_evaluation.pd.read_parquet")
def test_bandit_evaluation_run(
    mock_pd_read_parquet,
    mock_round_nearest_multiple,
    mock_os_makedirs,
    mock_visualize_line_plot,
    fixture_bandit_evaluation,
):
    """Test the Bandit Evaluation run method."""
    data_dict, _ = next(mock_generator())
    mock_pd_read_parquet.return_value = pd.DataFrame([data_dict])
    mock_round_nearest_multiple.return_value = 1

    fixture_bandit_evaluation.mlflow = MagicMock()
    fixture_bandit_evaluation.inputs = {"model_uri": True}
    fixture_bandit_evaluation.mlflow.pytorch.load_model.return_value = MockModel()
    fixture_bandit_evaluation.run()

    mock_pd_read_parquet.assert_called_once()
    fixture_bandit_evaluation.mlflow.pytorch.load_model.assert_called_once()
    mock_round_nearest_multiple.assert_called_once()
    mock_os_makedirs.assert_called_once()
    mock_visualize_line_plot.assert_called_once()
    fixture_bandit_evaluation.mlflow.log_artifact.assert_called_once()
    fixture_bandit_evaluation.logger.info.assert_called()
