"""Unit tests for preprocess_data.py."""
from unittest.mock import Mock, create_autospec, patch

import pandas as pd
import pytest

from template_pipelines.steps.time_series.preprocess_data import PreprocessDataStep


@pytest.fixture(scope="function")
def fixture_preprocessing():
    """Fixture for preprocess data step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        preprocessing = PreprocessDataStep()
        yield preprocessing


def test_preprocessing_run(fixture_preprocessing):
    """Test the run() method of the Preprocessing class.

    This test verifies that the run() method of the Preprocessing class correctly calls the generate_data() method,
    the get_output_fn() method, and checks the return value of get_output_fn().

    The test performs the following steps:
    1. Creates a mock DataFrame object.
    2. Sets the return value of the generate_data() method of the fixture_preprocessing object to the mock DataFrame.
    3. Sets the return value of the get_output_fn() method of the ioctx object to "/".
    4. Calls the run() method of the fixture_preprocessing object.
    5. Asserts that the generate_data() method was called.
    6. Asserts that the get_output_fn() method was called.
    7. Asserts that the get_output_fn() method was called only once.
    8. Asserts that the return value of the get_output_fn() method is "/".
    9. Asserts that the return value of the get_output_fn() method is not "/output".
    """
    mock_df = create_autospec(pd.DataFrame)
    fixture_preprocessing.generate_data = Mock(return_value=mock_df)
    fixture_preprocessing.ioctx.get_output_fn.return_value = "/"

    fixture_preprocessing.run()

    assert fixture_preprocessing.generate_data.called
    assert fixture_preprocessing.ioctx.get_output_fn.called
    assert fixture_preprocessing.ioctx.get_output_fn.call_count == 1
    assert fixture_preprocessing.ioctx.get_output_fn.return_value == "/"
    assert fixture_preprocessing.ioctx.get_output_fn.return_value != "/output"


def test_generate_data(fixture_preprocessing):
    """Test the generate_data() method of the PreprocessDataStep class.

    This test verifies that the generate_data() method of the PreprocessDataStep class correctly loads the CO2 data
    and returns a pandas Series.

    The test performs the following steps:
    1. Calls the generate_data() method of the fixture_preprocessing object.
    2. Asserts that the return value is an instance of pd.Series.
    """
    result = fixture_preprocessing.generate_data()
    assert isinstance(result, pd.Series)

    # assert results shape
    assert result.shape == (2284,)
