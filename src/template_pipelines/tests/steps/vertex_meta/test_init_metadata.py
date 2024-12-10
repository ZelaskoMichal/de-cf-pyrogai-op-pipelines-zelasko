"""Unit tests for template_pipelines/steps/vertex_meta/init_metadata.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from template_pipelines.steps.vertex_meta.init_metadata import InitMetadata


@pytest.fixture(scope="function")
def fixture_init_metadata():
    """Fixture for InitMetadata step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"), patch(
        "vertexai.init"
    ):
        im = InitMetadata()
        im.df = pd.DataFrame(
            {
                "column1": np.random.rand(10),
                "column2": np.random.rand(10),
                "column3": np.random.rand(10),
            }
        )
        im.outputs = {"params": "", "metrics": ""}
        yield im


def test_prepare_data(fixture_init_metadata):
    """Testing prepare_data method."""
    x_train, x_test, y_train, y_test = fixture_init_metadata._prepare_data(fixture_init_metadata.df)
    assert x_train.shape[1] == 2
    assert x_test.shape[1] == 2
    assert len(y_train) + len(y_test) == len(fixture_init_metadata.df)


def test_train_model(fixture_init_metadata):
    """Testing train_model method."""
    x_train = fixture_init_metadata.df[["column2", "column3"]]
    y_train = fixture_init_metadata.df["column1"]
    model = fixture_init_metadata._train_model(x_train, y_train)
    assert isinstance(model, LinearRegression)


def test_evaluate_model(fixture_init_metadata):
    """Testing evaluate_model method."""
    x_train = fixture_init_metadata.df[["column2", "column3"]]
    y_train = fixture_init_metadata.df["column1"]
    model = fixture_init_metadata._train_model(x_train, y_train)
    metrics, params, y_pred = fixture_init_metadata._evaluate_model(model, x_train, y_train)
    assert "mse" in metrics
    assert "mae" in metrics
    assert "r_squared" in metrics
    assert "coeff_0" in params
    assert "coeff_1" in params
    assert "intercept" in params


def test_generate_random_dataframe(fixture_init_metadata):
    """Testing generate_random_dataframe method."""
    df = fixture_init_metadata.generate_random_dataframe(num_rows=10, columns=["a", "b", "c"])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert list(df.columns) == ["a", "b", "c"]


@patch("joblib.dump")
def test_save_model(mock_dump, fixture_init_metadata):
    """Testing _save_model method."""
    fixture_init_metadata._save_model("model")
    mock_dump.assert_called_once()


@patch("template_pipelines.steps.vertex_meta.init_metadata.InitMetadata._plot_preds")
@patch("template_pipelines.steps.vertex_meta.init_metadata.InitMetadata._prepare_data")
@patch("template_pipelines.steps.vertex_meta.init_metadata.InitMetadata._train_model")
@patch("template_pipelines.steps.vertex_meta.init_metadata.InitMetadata._evaluate_model")
@patch("template_pipelines.steps.vertex_meta.init_metadata.InitMetadata._save_model")
def test_run(
    mock_save_model,
    mock_evaluate_model,
    mock_train_model,
    mock_prepare_data,
    mock_plot_preds,
    fixture_init_metadata,
):
    """Testing run method."""
    mock_model = MagicMock()
    mock_model.coef_ = "mock_coef"
    mock_model.intercept_ = "mock_intercept"
    mock_prepare_data.return_value = ("x_train", "x_test", "y_train", "y_test")
    mock_train_model.return_value = mock_model
    mock_evaluate_model.return_value = ("metrics", "params", "y_pred")
    fixture_init_metadata.run()
    mock_prepare_data.assert_called_once()
    mock_plot_preds.assert_called_once_with("y_test", "y_pred")
    mock_train_model.assert_called_once_with("x_train", "y_train")
    mock_evaluate_model.assert_called_once_with(mock_model, "x_test", "y_test")
    mock_save_model.assert_called_once_with(mock_model)
    assert fixture_init_metadata.outputs["metrics"] == "metrics"
    assert fixture_init_metadata.outputs["params"] == "params"
