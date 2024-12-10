"""Unit tests for template_pipelines/steps/iris_1_generate_data.py."""

from unittest.mock import Mock, create_autospec

import pandas as pd
import pytest

from aif.pyrogai.tests.conftest import mock_step_env
from template_pipelines.steps.ml_iris.iris_1_generate_data import GenerateDataStep


# This is other example of testing step from your pipeline. It uses mock from pyrogai conftest file.
@pytest.fixture
def fixture_generate_data_step(request):
    """Fixture for step."""
    with mock_step_env(request):
        yield GenerateDataStep()


@pytest.mark.parametrize(
    "fixture_generate_data_step",
    (
        {
            "platform": "Local",
            "config_module": "template_pipelines.config",
            "pipeline_name": "ml_iris",
        },
    ),
    indirect=True,
)
def test_generate_data_step_generate_dataframe(fixture_generate_data_step):
    """Test generate_dataframe()."""
    num_rows = 200

    df = fixture_generate_data_step.generate_dataframe(num_rows)

    assert df.shape == (num_rows, 5)
    assert list(df.columns) == [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class",
    ]
    assert isinstance(df, pd.DataFrame)


@pytest.mark.parametrize(
    "fixture_generate_data_step",
    (
        {
            "platform": "Local",
            "config_module": "template_pipelines.config",
            "pipeline_name": "ml_iris",
        },
    ),
    indirect=True,
)
def test_generate_data_step_run(fixture_generate_data_step):
    """Test run()."""
    mock_df = create_autospec(pd.DataFrame)
    fixture_generate_data_step.generate_dataframe = Mock(return_value=mock_df)
    fixture_generate_data_step.ioctx.get_output_fn.return_value = "/"

    fixture_generate_data_step.run()

    fixture_generate_data_step.ioctx.get_output_fn.assert_called_once_with("uploaded_data.parquet")
    mock_df.to_parquet.assert_called_once_with("/")
