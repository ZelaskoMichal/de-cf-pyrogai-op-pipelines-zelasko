"""Transformer class tests."""
import pandas as pd

from template_pipelines.steps.optimization.preprocessing.transformers import StocksTransformer


def test_calc_expected_returns():
    """Test that expected returns transformation succeeds."""
    # prepare
    t = StocksTransformer()
    input_df = pd.DataFrame([[10, 1.1]], columns=["avg_return", "region_index"])

    # execute
    result = t.calc_expected_returns(input_df)

    # assert, that expected_return was added and have expected value
    assert result["expected_return"][0] == 11


def test_merge_input_tables():
    """Verify that merging transformation succeeds."""
    # prepare
    t = StocksTransformer()

    raw_data = {
        "stocks": pd.DataFrame(
            [["IT", "EMEA"], ["FINANCE", "LAC"]], columns=["industry", "region"]
        ),
        "regions": pd.DataFrame([["EMEA", 1.1], ["LAC", 1.5]], columns=["region", "region_index"]),
        "industries": pd.DataFrame(
            [["IT", 7], ["FINANCE", 30]], columns=["industry", "avg_return"]
        ),
    }
    expected_result = pd.DataFrame(
        [["IT", "EMEA", 1.1, 7], ["FINANCE", "LAC", 1.5, 30]],
        columns=["industry", "region", "region_index", "avg_return"],
    )

    # execute
    result = t.merge_input_tables(raw_data)

    # assert
    pd.testing.assert_frame_equal(result, expected_result)


def test_filter_columns():
    """Verify that redundant columns are filtered out."""
    # prepare
    t = StocksTransformer()
    input_df = pd.DataFrame(
        [["IT", 1.1, "1", "EMEA", "abc"]],
        columns=["name", "expected_return", "is_risky", "region", "redundant_col"],
    )

    # execute
    result = t.filter_columns(input_df)

    # assert, that unexpected columns are removed
    expected_cols = set(input_df.columns).difference(set(["redundant_col"]))
    assert set(result.columns) == expected_cols
