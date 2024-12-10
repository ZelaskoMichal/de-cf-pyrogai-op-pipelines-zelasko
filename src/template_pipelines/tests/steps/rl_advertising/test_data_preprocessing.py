"""Tests for data preprocessing."""

import logging
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from template_pipelines.steps.rl_advertising.data_preprocessing import Preprocessing


@pytest.fixture(scope="function")
def fixture_preprocessing():
    """Fixture for the Preprocessing step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        pp = Preprocessing()
        yield pp


@pytest.fixture(scope="function")
def test_data():
    """Fixture for providing test data."""
    sample = pd.DataFrame(
        {
            "education": ["preschool", "some-college", "doctorate"],
            "native_country": ["united-states", "poland", "vietnam"],
            "workclass": ["never-worked", "without-pay", "?"],
            "occupation": ["?", "other-service", "clerk"],
            "age": [26, 27, 28],
        }
    )
    yield sample


def test_transform_categorical(fixture_preprocessing):
    """Test transform_categorical."""
    df = pd.DataFrame({"col1": [1, 2], "col2": ["Value1", " value2"]})

    res = fixture_preprocessing.transform_categorical(df)

    assert res["col2"].tolist() == ["value1", "value2"]


def test_reduce_categories(fixture_preprocessing, test_data):
    """Test reduce_categories."""
    expected = pd.DataFrame(
        {
            "education": ["elementary", "undergraduate", "graduate"],
            "native_country": ["us", "non-us", "non-us"],
            "workclass": 3 * ["undefined"],
        }
    )
    columns = expected.columns

    res = fixture_preprocessing.reduce_categories(test_data[columns])

    pd.testing.assert_frame_equal(res, expected)


def test_create_categories(fixture_preprocessing, test_data):
    """Test create_categories."""
    expected = pd.DataFrame(
        {
            "occupation": ["other", "other", "clerk"],
            "education": ["preschool", "some-college", "doctorate"],
        }
    )
    columns = expected.columns

    res = fixture_preprocessing.create_categories(test_data[columns])

    pd.testing.assert_frame_equal(res, expected)


def test_one_hot_encode(fixture_preprocessing, test_data):
    """Test one_hot_encode."""
    expected = pd.DataFrame(
        {
            "occupation_?": [1.0, 0.0, 0.0],
            "occupation_other-service": [0.0, 1.0, 0.0],
            "occupation_clerk": [0.0, 0.0, 1.0],
            "age": [26, 27, 28],
        }
    )
    res = fixture_preprocessing.one_hot_encode(test_data[["occupation", "age"]], "age")

    pd.testing.assert_frame_equal(res[expected.columns], expected)


def test_split_data(fixture_preprocessing, test_data):
    """Test split_data."""
    res = fixture_preprocessing.split_data(test_data, 0.8)

    assert isinstance(res, dict)
    assert list(res.keys()) == ["train_df", "test_df"]
    assert len(res["train_df"]) > len(res["test_df"])
    for value in res.values():
        assert isinstance(value, pd.DataFrame)


@patch("template_pipelines.steps.rl_advertising.data_preprocessing.pd.read_parquet")
def test_preprocessing_run(mock_pd_read_parquet, fixture_preprocessing, test_data):
    """Test the Preprocessing run method."""
    mock_pd_read_parquet.return_value = test_data
    fixture_preprocessing.logger = logging
    fixture_preprocessing.inputs = MagicMock()
    fixture_preprocessing.config = {"rl_advertising": {"train_size": 0.8}}

    with TestCase().assertLogs(level="INFO") as log:
        fixture_preprocessing.run()
    last_record = log.records[-1]
    msg = last_record.msg

    mock_pd_read_parquet.assert_called_once()
    assert msg == "Preprocessing is done."
