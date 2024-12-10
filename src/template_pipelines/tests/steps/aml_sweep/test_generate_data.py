"""Unit tests for template_pipelines/steps/anl_sweep/generate_data.py."""

from unittest.mock import Mock, call, create_autospec, patch

import pandas as pd
import pytest

from template_pipelines.steps.aml_sweep.generate_data import GenerateData


@pytest.fixture(scope="function")
def fixture_step():
    """Fixture for the step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        step = GenerateData()
        yield step


def test__generate_datafram(fixture_step):
    """Test step._generate_dataframe()."""
    num_rows = 17

    df = fixture_step._generate_dataframe(num_rows)

    assert df.shape == (num_rows, 5)
    assert list(df.columns) == [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]
    assert isinstance(df, pd.DataFrame)


def test__process_data(fixture_step):
    """Test step._process_data()."""
    num_rows = 20
    df = fixture_step._generate_dataframe(num_rows)
    pre_cols = df.columns

    train, test, enc = fixture_step._process_data(df)
    assert len(train) == 16
    assert all(train.columns == pre_cols)
    assert len(test) == 4
    assert all(test.columns == pre_cols)
    assert enc is not None


@patch("template_pipelines.steps.aml_sweep.generate_data.pickle.dump")
def test_generate_data_step_run(mock_pkl_dump, fixture_step):
    """Test run()."""
    mock_df = create_autospec(pd.DataFrame)
    fixture_step._generate_dataframe = Mock(return_value=mock_df)
    fixture_step._process_data = Mock(side_effect=lambda x: (x, x, x))
    fixture_step.ioctx.get_output_fn.return_value = "/"
    with patch("builtins.open"):
        fixture_step.run()

    fixture_step.ioctx.get_output_fn.assert_has_calls(
        [call("data/train.parquet"), call("data/test.parquet"), call("encoder.pkl")]
    )
    assert mock_df.to_parquet.call_count == 2
    assert mock_pkl_dump.call_count == 1
