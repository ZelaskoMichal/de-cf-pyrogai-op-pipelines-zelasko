"""Unit tests for opinionated_pipelines/steps/data_preprocessing.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from template_pipelines.steps.ml_training.data_preprocessing import Preprocessing


@pytest.fixture(scope="function")
def fixture_preprocessing():
    """Fixture for Preprocessing step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        pp = Preprocessing()

        yield pp


def test_preprocessing_tables_into_dict(fixture_preprocessing):
    """Test tables_into_dict."""
    fixture_preprocessing.inputs = {"campaigns": "", "coupon_item_mapping": ""}

    with patch("template_pipelines.steps.ml_training.data_preprocessing.pd") as mock_pd:
        res = fixture_preprocessing.tables_into_dict()

        assert mock_pd.read_parquet.call_count == 2
        assert isinstance(res, dict)
        assert list(res.keys()) == ["campaigns", "coupon_item_mapping"]


def test_preprocessing_transform_coupon_redemption(fixture_preprocessing):
    """Test transform_coupon_redemption."""
    tables = {
        "coupon_redemption": pd.DataFrame(
            {
                "redemption_status": [1, np.nan, 2, np.nan, 3],
                "other_column": ["a", "b", "c", "d", "e"],
            }
        )
    }

    fixture_preprocessing.transform_coupon_redemption(tables)

    expected_df = pd.DataFrame(
        {"redemption_status": [1.0, 2.0, 3.0], "other_column": ["a", "c", "e"]},
        index=[0, 2, 4],
    )

    pd.testing.assert_frame_equal(tables["coupon_redemption"], expected_df)


def test_preprocessing_transform_customer_transactions(fixture_preprocessing):
    """Test transform_customer_transactions."""
    tables = {
        "customer_transactions": pd.DataFrame(
            {
                "date": ["2022-01-03", "2022-01-01", "2022-01-02"],
                "other_column": ["a", "b", "c"],
            }
        )
    }

    fixture_preprocessing.transform_customer_transactions(tables)

    expected_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03"]),
            "other_column": ["b", "c", "a"],
        }
    )

    pd.testing.assert_frame_equal(tables["customer_transactions"], expected_df)


def test_preprocessing_transform_campaigns(fixture_preprocessing):
    """Test transform_campaigns."""
    tables = {
        "campaigns": pd.DataFrame(
            {
                "start_date": ["2022-01-03", "2022-01-01", "2022-01-02"],
                "end_date": ["2022-02-03", "2022-02-01", "2022-02-02"],
                "other_column": ["a", "b", "c"],
            }
        )
    }

    fixture_preprocessing.transform_campaigns(tables)

    expected_df = pd.DataFrame(
        {
            "start_date": pd.to_datetime(["2022-01-03", "2022-01-01", "2022-01-02"]),
            "end_date": pd.to_datetime(["2022-02-03", "2022-02-01", "2022-02-02"]),
            "other_column": ["a", "b", "c"],
        }
    )

    pd.testing.assert_frame_equal(tables["campaigns"], expected_df)


def test_preprocessing_rename_and_generalize_columns(fixture_preprocessing):
    """Test rename_and_generalize_columns."""
    tables = {
        "customer_demographics": pd.DataFrame(
            {
                "marital_status": ["Married", "Single", "Married", np.nan],
                "other_column": ["a", "b", "c", "d"],
            }
        )
    }

    fixture_preprocessing.rename_and_generalize_columns(tables)

    expected_df = pd.DataFrame(
        {
            "relationship_status": [
                "In relationship",
                "Single",
                "In relationship",
                np.nan,
            ],
            "other_column": ["a", "b", "c", "d"],
        }
    )

    pd.testing.assert_frame_equal(tables["customer_demographics"], expected_df)


def test_preprocessing_fill_missing_values(fixture_preprocessing):
    """Test fill_missing_values."""
    tables = {
        "customer_demographics": pd.DataFrame(
            {
                "relationship_status": [np.nan, np.nan, "In relationship"],
                "family_size": ["1", "2", "3"],
                "no_of_children": [np.nan, np.nan, "1"],
            }
        )
    }

    fixture_preprocessing.fill_missing_values(tables)

    expected_df = pd.DataFrame(
        {
            "relationship_status": ["Single", "In relationship", "In relationship"],
            "family_size": ["1", "2", "3"],
            "no_of_children": [np.nan, np.nan, "1"],
        }
    )

    pd.testing.assert_frame_equal(tables["customer_demographics"], expected_df)


def test_preprocessing_create_additional_columns(fixture_preprocessing):
    """Test create_additional_columns."""
    tables = {
        "customer_demographics": pd.DataFrame(
            {
                "family_size": ["1", "2+", "1", "3+"],
                "no_of_children": ["1", "2+", np.nan, "3+"],
                "other_column": ["a", "b", "c", "d"],
            }
        )
    }

    fixture_preprocessing.create_additional_columns(tables)

    expected_df = pd.DataFrame(
        {
            "family_size": ["1", "2+", "1", "3+"],
            "no_of_children": ["1", "2+", np.nan, "3+"],
            "other_column": ["a", "b", "c", "d"],
            "family_size_int": [1.0, 2.0, 1.0, 3.0],
            "no_of_children_int": [1.0, 2.0, np.nan, 3.0],
        }
    )

    pd.testing.assert_frame_equal(tables["customer_demographics"], expected_df)


def test_preprocessing_single_or_in_relationship(fixture_preprocessing):
    """Test single_or_in_relationship."""
    tables = {
        "customer_demographics": pd.DataFrame(
            {
                "family_size": ["1", "2", "3+", "2"],
                "family_size_int": [1.0, 2.0, 3.0, 2.0],
                "no_of_children": ["0", "1+", np.nan, "0"],
                "no_of_children_int": [0.0, 1.0, np.nan, 0.0],
                "relationship_status": [None, None, None, None],
                "other_column": ["a", "b", "c", "d"],
            }
        )
    }

    fixture_preprocessing.single_or_in_relationship(tables)

    expected_df = pd.DataFrame(
        {
            "family_size": ["1", "2", "3+", "2"],
            "family_size_int": [1.0, 2.0, 3.0, 2.0],
            "no_of_children": ["0", "1+", np.nan, "0"],
            "no_of_children_int": [0.0, 1.0, np.nan, 0.0],
            "relationship_status": [
                "Single",
                "Single",
                None,
                "In relationship",
            ],
            "other_column": ["a", "b", "c", "d"],
        }
    )

    pd.testing.assert_frame_equal(tables["customer_demographics"], expected_df)


def test_preprocessing_drop_unneeded_columns(fixture_preprocessing):
    """Test drop_unneeded_columns."""
    tables = {
        "customer_demographics": pd.DataFrame(
            {
                "family_size_int": [1.0, 2.0, 3.0, 2.0],
                "no_of_children_int": [0.0, 1.0, np.nan, 0.0],
                "relationship_status": [
                    "Single",
                    "In relationship",
                    None,
                    "In relationship",
                ],
                "other_column": ["a", "b", "c", "d"],
            }
        )
    }

    fixture_preprocessing.drop_unneeded_columns(tables)

    expected_df = pd.DataFrame(
        {
            "relationship_status": [
                "Single",
                "In relationship",
                None,
                "In relationship",
            ],
            "other_column": ["a", "b", "c", "d"],
        }
    )

    pd.testing.assert_frame_equal(tables["customer_demographics"], expected_df)


def test_preprocessing_set_no_of_children(fixture_preprocessing):
    """Test set_no_of_children."""
    tables = {
        "customer_demographics": pd.DataFrame(
            {
                "family_size": ["1", "2", "3", "2", "1", "2"],
                "relationship_status": [
                    None,
                    "In relationship",
                    None,
                    "Single",
                    "Single",
                    None,
                ],
                "no_of_children": [None, None, "1", None, None, None],
                "other_column": ["a", "b", "c", "d", "e", "f"],
            }
        )
    }

    fixture_preprocessing.set_no_of_children(tables)

    expected_df = pd.DataFrame(
        {
            "family_size": ["1", "2", "3", "2", "1", "2"],
            "relationship_status": [
                None,
                "In relationship",
                None,
                "Single",
                "Single",
                None,
            ],
            "no_of_children": ["0", "0", "1", "1", "0", None],
            "other_column": ["a", "b", "c", "d", "e", "f"],
        }
    )

    pd.testing.assert_frame_equal(tables["customer_demographics"], expected_df)


@patch.object(Preprocessing, "set_no_of_children")
@patch.object(Preprocessing, "drop_unneeded_columns")
@patch.object(Preprocessing, "single_or_in_relationship")
@patch.object(Preprocessing, "create_additional_columns")
@patch.object(Preprocessing, "fill_missing_values")
@patch.object(Preprocessing, "rename_and_generalize_columns")
def test_preprocessing_transform_customer_demographics_calls(
    mock_rename,
    mock_fill,
    mock_create,
    mock_single,
    mock_drop,
    mock_set,
    fixture_preprocessing,
):
    """Test transform_customer_demographics calls."""
    tables = {}
    fixture_preprocessing.transform_customer_demographics(tables)

    mock_rename.assert_called_once_with(tables)
    mock_fill.assert_called_once_with(tables)
    mock_create.assert_called_once_with(tables)
    mock_single.assert_called_once_with(tables)
    mock_drop.assert_called_once_with(tables)
    mock_set.assert_called_once_with(tables)


@patch.object(Preprocessing, "transform_customer_demographics")
@patch.object(Preprocessing, "transform_campaigns")
@patch.object(Preprocessing, "transform_customer_transactions")
@patch.object(Preprocessing, "transform_coupon_redemption")
def test_preprocessing_data_transformation_calls(
    mock_coupon, mock_transactions, mock_campaigns, mock_demo, fixture_preprocessing
):
    """Test data_transformation calls."""
    tables = {}
    fixture_preprocessing.data_transformation(tables)

    mock_coupon.assert_called_once_with(tables)
    mock_transactions.assert_called_once_with(tables)
    mock_campaigns.assert_called_once_with(tables)
    mock_demo.assert_called_once_with(tables)


@patch("template_pipelines.steps.ml_training.data_preprocessing.save_tables")
@patch.object(Preprocessing, "label_encoding")
@patch.object(Preprocessing, "data_transformation")
@patch.object(Preprocessing, "tables_into_dict")
def test_preprocessing_run(
    mock_tables_into_dict,
    mock_data_transformation,
    mock_label_encoding,
    mock_save_tables,
    fixture_preprocessing,
):
    """Test run method."""
    mock_tables_into_dict.return_value = {}
    fixture_preprocessing.ioctx = MagicMock()

    fixture_preprocessing.run()

    mock_tables_into_dict.assert_called_once()
    mock_data_transformation.assert_called_once_with(tables={})
    mock_label_encoding.assert_called_once_with({})
    mock_save_tables.assert_called_once_with(
        {},
        fixture_preprocessing.ioctx.get_output_fn("preprocessed"),
        fixture_preprocessing.logger,
    )
