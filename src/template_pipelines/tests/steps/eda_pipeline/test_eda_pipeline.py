"""Unit tests for the EdaExample class."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from template_pipelines.steps.eda_pipeline.eda_pipeline import EdaExample


@pytest.fixture(scope="function")
def fixture_eda_example():
    """Fixture for EdaExample step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        eda_example = EdaExample()
        eda_example.logger = MagicMock()
        eda_example.config = {"eda_tool": "pandas_profiling"}
        yield eda_example


def test_load_iris_data(fixture_eda_example):
    """Test load_iris_data method."""
    data = fixture_eda_example.load_iris_data()

    assert isinstance(data, pd.DataFrame)
    assert "target" in data.columns
    assert len(data) > 0
    fixture_eda_example.logger.info.assert_called_with("Loading Iris dataset...")


@patch("template_pipelines.steps.eda_pipeline.eda_pipeline.EDAToolkit")
def test_perform_eda(mock_eda_toolkit_class, fixture_eda_example):
    """Test perform_eda method."""
    data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

    fixture_eda_example.perform_eda(data, tool="pandas_profiling")

    mock_eda_toolkit_class.assert_called_with(data, fixture_eda_example.logger)
    mock_eda_toolkit_class.return_value.generate_report.assert_called_with("pandas_profiling")


@patch.object(EdaExample, "perform_eda")
def test_run(mock_perform_eda, fixture_eda_example):
    """Test run method."""
    fixture_eda_example.run()

    fixture_eda_example.logger.info.assert_any_call("Performing EDA...")
    mock_perform_eda.assert_called()

    args, kwargs = mock_perform_eda.call_args
    data_arg = args[0]
    tool_arg = kwargs.get("tool")

    assert isinstance(data_arg, pd.DataFrame)
    assert tool_arg == fixture_eda_example.config["eda_tool"]
    fixture_eda_example.logger.info.assert_any_call("EDA step completed.")
