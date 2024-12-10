"""Utilities to formulation tests."""

from typing import List, Optional

import pandas as pd

from dnalib.optimization_models.xpress_utils.testing import assert_columns_exist, assert_df_length


def assert_df(
    df: pd.DataFrame,
    cols: List[str],
    expected_df_len: Optional[int] = None,
    check_nulls: Optional[bool] = True,
) -> None:
    """Helper to validate generation of dataframe."""
    # start by checking if cols exist
    assert_columns_exist(df, cols)

    # check length of dataframe
    if expected_df_len:
        assert_df_length(df, expected_df_len=expected_df_len)

    # check if there are null values in variable cols
    if check_nulls:
        col_with_null = []
        for col in cols:
            if not df[col].notnull().all():
                col_with_null.append(col)
        assert (
            len(col_with_null) == 0
        ), f"The following variable columns have null values: {col_with_null}. Tip: you might want to have 0s instead."
