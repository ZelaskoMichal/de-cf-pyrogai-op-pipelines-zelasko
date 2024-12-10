"""Tests for data preprocessing."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from template_pipelines.steps.gen_ai_product_opt.data_preprocessing import Preprocessing


@pytest.fixture(scope="function")
def fixture_preprocessing():
    """Fixture for the Preprocessing step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        pp = Preprocessing()
        yield pp


@pytest.fixture(scope="function")
def test_data():
    """Fixture for providing test data."""
    sample = {
        "advertised_products": pd.DataFrame(
            {
                "product_id": [1, 2],
                "product_category": ["a", "b"],
                "title": ["title_a", "title_b"],
                "desc_1": ["desc_1a", "desc_1b"],
                "desc_2": ["desc_2a", "desc_2b"],
            }
        ),
        "product_keywords": pd.DataFrame(
            {"product_id": [1, 2], "keywords": ["key1a,key1b", "key2a,key2b"]}
        ),
    }
    yield sample


def test_read_data_into_dict(fixture_preprocessing):
    """Test read_data_into_dict."""
    fixture_preprocessing.inputs = {
        "advertised_products": "advertised_products.parquet",
        "product_keywords": "product_keywords.parquet",
    }

    with patch("template_pipelines.steps.gen_ai_product_opt.data_preprocessing.pd") as mock_pd:
        res = fixture_preprocessing.read_data_into_dict()

        assert mock_pd.read_parquet.call_count == 2
        assert isinstance(res, dict)
        assert res.keys() == fixture_preprocessing.inputs.keys()


def test_set_datatype(fixture_preprocessing, test_data):
    """Test set_datatype."""
    res = fixture_preprocessing.set_datatype(test_data, str)
    datatypes = np.unique([df.dtypes.unique()[0] for df in res.values()])

    assert datatypes == [np.dtype(object)]


def test_combine_text_columns(fixture_preprocessing, test_data):
    """Test combine_text_columns."""
    df = test_data["advertised_products"]
    res = fixture_preprocessing.combine_text_columns(df, "desc_", ";", "new_column")
    expected_series = pd.Series(data=["desc_1a;desc_2a", "desc_1b;desc_2b"], name="new_column")

    pd.testing.assert_series_equal(res["new_column"], expected_series)


def test_transform_text(fixture_preprocessing, test_data):
    """Test transform_text."""
    df = test_data["product_keywords"]
    res = fixture_preprocessing.transform_text(df, "keywords")
    expected_series = pd.Series(data=["key1a,key1b", "key2a,key2b"], name="keywords")

    pd.testing.assert_series_equal(res["keywords"], expected_series)


def test_left_join_all(fixture_preprocessing, test_data):
    """Test left_join_all."""
    res = fixture_preprocessing.left_join_all(test_data, "product_id")
    expected_columns = ["product_id", "product_category", "title", "desc_1", "desc_2", "keywords"]

    assert list(res.columns) == expected_columns


@patch("template_pipelines.steps.gen_ai_product_opt.data_preprocessing.pd.DataFrame.to_parquet")
@patch.object(Preprocessing, "read_data_into_dict")
def test_preprocessing_run(
    mock_read_data_into_dict, mock_to_parquet, fixture_preprocessing, test_data
):
    """Test the Preprocessing run method."""
    mock_read_data_into_dict.return_value = test_data
    mock_to_parquet.return_value = True
    fixture_preprocessing.run()

    mock_read_data_into_dict.assert_called_once()
    mock_to_parquet.assert_called_once()
