"""Tests for toolkit."""

import math
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn

from template_pipelines.utils.rl_advertising import toolkit


class MockModel(nn.Module):
    """A mock neural network model for testing purposes."""

    def __init__(self):
        """Initialize the MockModel instance."""
        super().__init__()

    def forward(self, x):
        """Perform a forward pass through the network."""
        return torch.tensor([1.0])


class TestBayesianNN(TestCase):
    """Test cases for the BayesianNN class."""

    def setUp(self):
        """Set up a test environment before each test method is executed."""
        self.n_input = 100
        self.dropout = 0.1
        self.bayesian_nn = toolkit.BayesianNN(self.n_input, self.dropout)
        self.bayesian_nn_no_dropout = toolkit.BayesianNN(self.n_input, 0)

    def tearDown(self):
        """Remove the test environment after each test method is executed."""

    def test__init__(self):
        """Test initialized layers and properties."""
        self.assertIsInstance(self.bayesian_nn.fc1, nn.Linear)
        self.assertIsInstance(self.bayesian_nn.fc2, nn.Linear)
        self.assertIsInstance(self.bayesian_nn.fc3, nn.Linear)
        self.assertIsInstance(self.bayesian_nn.relu, nn.ReLU)
        self.assertIsInstance(self.bayesian_nn.sigmoid, nn.Sigmoid)
        self.assertIsInstance(self.bayesian_nn.dropout, nn.Dropout)
        self.assertIs(self.bayesian_nn.use_dropout, True)
        self.assertIs(self.bayesian_nn_no_dropout.use_dropout, False)

    def test_forward(self):
        """Test the forward pass of the BayesianNN model."""
        input_tensor = torch.randn(10, self.n_input)
        output = self.bayesian_nn(input_tensor)

        self.assertEqual(output.shape, (10, 1))
        self.assertTrue(torch.all(output >= 0) & torch.all(output <= 1))

    def test_dropout_usage(self):
        """Test that the dropout is used correctly."""
        x = torch.ones(1, self.n_input)

        self.bayesian_nn_no_dropout.train()
        no_dropout_output_train = self.bayesian_nn_no_dropout(x)

        self.bayesian_nn_no_dropout.eval()
        no_dropout_output_eval = self.bayesian_nn_no_dropout(x)

        self.bayesian_nn.train()
        dropout_output_train = self.bayesian_nn(x)

        self.bayesian_nn.eval()
        dropout_output_eval = self.bayesian_nn(x)

        self.assertTrue(torch.equal(no_dropout_output_train, no_dropout_output_eval))
        self.assertFalse(torch.equal(dropout_output_train, dropout_output_eval))


def test_build_model():
    """Test build_model."""
    n_input = 100
    dropout = 0.1
    model, optimizer, loss = toolkit.build_model(n_input, dropout)

    assert isinstance(model, nn.Module)
    assert model.use_dropout is True
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert isinstance(loss, nn.Module)


def test_train_per_epoch():
    """Test train_per_epoch."""
    n_input = 100
    dropout = 0.1
    model, optimizer, loss = toolkit.build_model(n_input, dropout)
    x = abs(torch.randn(10, n_input))
    y = torch.randn(10)
    y = (y - y.min()) / (y.max() - y.min())
    initial_params = list(model.parameters())[0].clone()

    loss = toolkit.train_per_epoch(x, y, model, loss, optimizer)
    updated_params = list(model.parameters())[0].clone()

    for param in model.parameters():
        if isinstance(param.grad, torch.Tensor):
            assert torch.equal(param.grad, torch.zeros(param.grad.shape))
        else:
            assert param.grad is None
    assert isinstance(loss, float)
    assert torch.equal(initial_params, updated_params) is False


def test_update_model():
    """Test update_model."""
    n_input = 100
    dropout = 0.1
    model, optimizer, loss = toolkit.build_model(n_input, dropout)
    x = abs(np.random.randn(10, 1, n_input))
    y = np.random.randn(10)
    y = (y - y.min()) / (y.max() - y.min())
    initial_loss = float("inf")
    initial_params = list(model.parameters())[0].clone()

    with TestCase().assertLogs(level="INFO") as log:
        toolkit.update_model(x, y, model, optimizer, loss)

    updated_params = list(model.parameters())[0].clone()

    assert torch.equal(initial_params, updated_params) is False
    assert len(log.records) == 10
    for record in log.records:
        final_loss = float(record.msg.split()[-1])
        assert "Training loss:" in record.msg
    assert final_loss < initial_loss


def test_enable_dropout():
    """Test enable_dropout."""
    n_input = 100
    dropout = 0.1
    model = toolkit.BayesianNN(n_input, dropout)

    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            assert m.training is False

    toolkit.enable_dropout(model)
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            assert m.training is True

    for m in model.modules():
        if not m.__class__.__name__.startswith("Dropout"):
            assert m.training is False


def test_simulate_ad_inventory():
    """Test simulate_ad_inventory."""
    expected_ads = ["elementary", "middle", "hs-grad", "undergraduate", "graduate"]
    res = toolkit.simulate_ad_inventory()

    assert len(res) <= len(expected_ads)
    assert len(set(res).intersection(expected_ads)) == len(res)


def test_simulate_ad_click_probs():
    """Test simulate_ad_click_probs."""
    expected_ads = ["elementary", "middle", "hs-grad", "undergraduate", "graduate"]
    res = toolkit.simulate_ad_click_probs()

    assert isinstance(res, dict)
    assert set(res.keys()) == set(expected_ads)

    for key in res:
        assert set(res[key].keys()) == set(expected_ads)

    for key1 in res:
        for key2 in res[key1]:
            assert res[key1][key2] >= 0 and res[key1][key2] <= 1


@patch("numpy.random.uniform")
def test_simulate_click(mock_uniform):
    """Test simulate click."""
    ad_click_probs = toolkit.simulate_ad_click_probs()
    ad = "graduate"
    user = {"education": ad}

    mock_uniform.return_value = 0.25
    click_res = toolkit.simulate_click(ad_click_probs, user, ad)

    mock_uniform.return_value = 0.95
    no_click_res = toolkit.simulate_click(ad_click_probs, user, ad)

    assert click_res == 1
    assert no_click_res == 0


def test_select_ad():
    """Test select_ad."""
    model = MockModel()
    context = [1, 2, 3, 4]
    expected_x = np.array([context + [0, 0, 0, 0, 1]], dtype=np.float32)
    expected_ad = "graduate"
    ad_inventory = [expected_ad]

    selected_ad_res, selected_x_res = toolkit.select_ad(model, context, ad_inventory)

    assert selected_ad_res == expected_ad
    assert np.array_equal(selected_x_res, expected_x)


def test_one_hot_ad():
    """Test one_hot_ad."""
    ad = "graduate"
    expected = [0, 0, 0, 0, 1]
    res = toolkit.one_hot_ad(ad)

    assert isinstance(res, list)
    assert res == expected


def test_generate_user():
    """Test generate_user."""
    data = pd.DataFrame(
        {"feat1": [1, 2, 3, 4, 5], "feat2": [6, 7, 8, 9, 10], "label": [1, 1, 1, 0, 0]}
    )
    feat_cols = data.columns[:-1]
    generator = toolkit.generate_user(data)
    user, context = next(generator)

    assert set(user.keys()) == set(data.keys())
    assert len(context) == len(feat_cols)
    assert set(context) == {user[feat] for feat in feat_cols}


def test_calculate_regret():
    """Test calculate_regret."""
    ad1 = "graduate"
    ad2 = "hs-grad"
    ad_selected = ad1
    ad_invetory = [ad1, ad2]
    user = {"education": ad1}
    ad_click_probs = {ad1: {ad1: 0.8}, ad2: {ad1: 0.4}}

    res = toolkit.calculate_regret(user, ad_invetory, ad_click_probs, ad_selected)

    assert math.isclose(res, 0.0)


@patch("template_pipelines.utils.rl_advertising.toolkit.os")
def test_save_tables(mock_os):
    """Test save_tables with multiple tables."""
    tables = {"df1": MagicMock(), "df2": MagicMock(), "df3": MagicMock()}
    output_directory = "output_dir"

    toolkit.save_tables(tables, output_directory)

    for table_name, mock_df in tables.items():
        mock_df.to_parquet.assert_called_once_with(
            (Path(output_directory) / (table_name + ".parquet")).as_posix()
        )

    mock_os.makedirs.assert_called_once_with(output_directory, exist_ok=True)


@patch("template_pipelines.utils.rl_advertising.toolkit.plt.savefig")
def test_visualize_line_plot(mock_plt_savefig):
    """Test visualize_line_plot."""
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    output_path = "test.png"
    mock_plt_savefig.return_value = None

    res = toolkit.visualize_line_plot(df, "x", "y", output_path)

    mock_plt_savefig.assert_called_once_with(output_path)
    assert isinstance(res, matplotlib.figure.Figure)
