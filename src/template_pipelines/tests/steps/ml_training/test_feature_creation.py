"""Unit tests for opinionated_pipelines/steps/feature_creation.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from template_pipelines.steps.ml_training.feature_creation import FeatureCreation


@pytest.fixture(scope="function")
def fixture_feature_creation():
    """Fixture for FeatureCreation step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        fc = FeatureCreation()

        yield fc


def test_feature_creation_get_campaign_duration(fixture_feature_creation):
    """Test for get_campaign_duration."""
    data = {
        "start_date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
        "end_date": pd.to_datetime(["2020-01-10", "2020-02-10", "2020-03-10"]),
    }
    tables = {"campaigns": pd.DataFrame(data)}

    fixture_feature_creation.get_campaign_duration(tables)

    assert "campaign_duration" in tables["campaigns"].columns
    pd.testing.assert_series_equal(
        tables["campaigns"]["campaign_duration"],
        pd.Series([9, 9, 9]),
        check_names=False,
    )
    assert "start_date" not in tables["campaigns"].columns
    assert "end_date" not in tables["campaigns"].columns


def test_feature_creation_get_unit_price_and_categorical_variable(
    fixture_feature_creation,
):
    """Test for get_unit_price_and_categorical_variable."""
    data = {
        "selling_price": pd.Series([100.0, 200.0, 300.0]),
        "quantity": pd.Series([10, 20, 30]),
        "other_discount": pd.Series([10.0, 20.0, 30.0]),
        "coupon_discount": pd.Series([0.0, 0.0, 5.0]),
    }
    tables = {"customer_transactions": pd.DataFrame(data)}
    fixture_feature_creation.get_unit_price_and_categorical_variable(tables)

    assert "price_after_discount" in tables["customer_transactions"].columns
    assert "discount" in tables["customer_transactions"].columns
    assert "price_before_discount" in tables["customer_transactions"].columns
    assert "counpon_used" in tables["customer_transactions"].columns

    pd.testing.assert_series_equal(
        tables["customer_transactions"]["price_after_discount"],
        pd.Series([10.0, 10.0, 10.0]),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        tables["customer_transactions"]["discount"],
        pd.Series([1.0, 1.0, 1.0]),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        tables["customer_transactions"]["price_before_discount"],
        pd.Series([9.0, 9.0, 9.0]),
        check_names=False,
    )
    assert tables["customer_transactions"]["counpon_used"].to_dict() == {
        0: 0,
        1: 0,
        2: 1,
    }


def test_feature_creation_get_customer_agg_trans(fixture_feature_creation):
    """Test for get_customer_agg_trans."""
    data = {
        "customer_id": pd.Series([1, 1, 2, 2]),
        "item_id": pd.Series([10, 20, 30, 40]),
        "quantity": pd.Series([1, 2, 3, 4]),
        "price_before_discount": pd.Series([100.0, 200.0, 300.0, 400.0]),
        "discount": pd.Series([10.0, 20.0, 30.0, 40.0]),
        "coupon_discount": pd.Series([5.0, 10.0, 15.0, 20.0]),
        "counpon_used": pd.Series([1, 0, 1, 0], dtype=np.int32),
        "dayofmonth": pd.Series([1, 2, 3, 4]),
        "dayofweek": pd.Series([1, 2, 2, 1]),
        "month": pd.Series([1, 1, 2, 2]),
    }
    tables = {"customer_transactions": pd.DataFrame(data)}

    result = fixture_feature_creation.get_customer_agg_trans(tables)

    df_test = pd.DataFrame(
        {
            "customer_id": {0: 1, 1: 2},
            "customer_no_of_unique_items": {0: 2, 1: 2},
            "customer_no_of_items": {0: 30, 1: 70},
            "customer_median_quantity": {0: 1.5, 1: 3.5},
            "customer_q1_quantity": {0: 1.25, 1: 3.25},
            "customer_q3_quantity": {0: 1.75, 1: 3.75},
            "customer_median_price": {0: 150.0, 1: 350.0},
            "customer_q1_price": {0: 125.0, 1: 325.0},
            "customer_q3_price": {0: 175.0, 1: 375.0},
            "customer_median_discount": {0: 15.0, 1: 35.0},
            "customer_q1_discount": {0: 12.5, 1: 32.5},
            "customer_q3_discount": {0: 17.5, 1: 37.5},
            "customer_median_coupon_discount": {0: 7.5, 1: 17.5},
            "customer_q1_coupon_discount": {0: 6.25, 1: 16.25},
            "customer_q3_coupon_discount": {0: 8.75, 1: 18.75},
            "customer_total_coupon_used": {0: 1, 1: 1},
            "customer_mode_dayofmonth": {0: 1, 1: 3},
            "customer_mode_dayofweek": {0: 1, 1: 1},
            "customer_mode_month": {0: 1, 1: 2},
        },
    )
    assert result.to_dict() == df_test.to_dict()


def test_feature_creation_get_coupon_trans(fixture_feature_creation):
    """Test for get_coupon_trans."""
    data_coupon_items = {
        "coupon_id": pd.Series([1, 1, 2, 2]),
        "item_id": pd.Series([10, 20, 30, 40]),
    }
    data_customer_transactions = {
        "customer_id": pd.Series([1, 1, 2, 2]),
        "item_id": pd.Series([10, 20, 30, 40]),
        "brand": pd.Series(["a", "b", "c", "d", "e"]),
        "brand_type": pd.Series(["x", "x", "y", "y", "x"]),
        "category": pd.Series(["cat1", "cat2", "cat3", "cat1", "cat2"]),
        "quantity": pd.Series([1, 2, 3, 4]),
        "price_before_discount": pd.Series([100.0, 200.0, 300.0, 400.0]),
        "discount": pd.Series([10.0, 20.0, 30.0, 40.0]),
        "coupon_discount": pd.Series([5.0, 10.0, 15.0, 20.0]),
        "counpon_used": pd.Series([1, 0, 1, 0], dtype=np.int32),
        "dayofmonth": pd.Series([1, 2, 3, 4]),
        "dayofweek": pd.Series([1, 2, 2, 1]),
        "month": pd.Series([1, 1, 2, 2]),
    }
    tables = {
        "coupon_items": pd.DataFrame(data_coupon_items),
        "customer_transactions": pd.DataFrame(data_customer_transactions),
    }
    result = fixture_feature_creation.get_coupon_trans(tables)
    df_test = pd.DataFrame(
        {
            "coupon_id": {0: 1, 1: 2},
            "coupon_no_of_item": {0: 2, 1: 2},
            "coupon_mode_brand": {0: "a", 1: "c"},
            "coupon_mode_brand_type": {0: "x", 1: "y"},
            "coupon_mode_category": {0: "cat1", 1: "cat1"},
            "coupon_no_of_customers": {0: 1, 1: 1},
            "coupon_q1_quantity": {0: 1.25, 1: 3.25},
            "coupon_q3_quantity": {0: 1.75, 1: 3.75},
            "coupon_median_price": {0: 150.0, 1: 350.0},
            "coupon_q1_price": {0: 125.0, 1: 325.0},
            "coupon_q3_price": {0: 175.0, 1: 375.0},
            "coupon_median_other_discount": {0: 15.0, 1: 35.0},
            "coupon_q1_other_discount": {0: 12.5, 1: 32.5},
            "coupon_q3_other_discount": {0: 17.5, 1: 37.5},
            "coupon_median_coupon_discount": {0: 7.5, 1: 17.5},
            "coupon_q1_coupon_discount": {0: 6.25, 1: 16.25},
            "coupon_q3_coupon_discount": {0: 8.75, 1: 18.75},
            "coupon_no_of_coupon_used": {0: 1.0, 1: 1.0},
        }
    )
    assert result.to_dict() == df_test.to_dict()


@patch("template_pipelines.steps.ml_training.feature_creation.plt")
@patch("template_pipelines.steps.ml_training.feature_creation.sns")
@patch("template_pipelines.steps.ml_training.feature_creation.os")
@patch("template_pipelines.steps.ml_training.feature_creation.left_join_all")
@patch.object(FeatureCreation, "get_coupon_trans")
@patch.object(FeatureCreation, "get_customer_agg_trans")
@patch.object(FeatureCreation, "get_unit_price_and_categorical_variable")
@patch.object(FeatureCreation, "get_campaign_duration")
@patch("template_pipelines.steps.ml_training.feature_creation.load_tables")
def test_fixture_creation_run(
    mock_load_tables,
    mock_get_campaign_duration,
    mock_get_unit_price_and_categorical_variable,
    mock_get_customer_agg_trans,
    mock_get_coupon_trans,
    mock_left_join_all,
    mock_os,
    mock_sns,
    mock_plt,
    fixture_feature_creation,
):
    """Test for run."""
    fixture_feature_creation.mlflow = MagicMock()
    mock_plt.subplots.return_value = MagicMock(), MagicMock()

    fixture_feature_creation.run()

    mock_load_tables.assert_called_once()
    mock_get_campaign_duration.assert_called_once()
    mock_get_unit_price_and_categorical_variable.assert_called_once()
    mock_get_customer_agg_trans.assert_called_once()
    mock_get_coupon_trans.assert_called_once()
    mock_left_join_all.assert_called_once()
    mock_os.makedirs.assert_called_once()
    mock_sns.heatmap.assert_called_once()
    mock_plt.savefig.assert_called_once()
    mock_plt.close.assert_called_once()
    fixture_feature_creation.mlflow.log_artifact.assert_called_once()
    fixture_feature_creation.logger.info.assert_called()
