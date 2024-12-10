"""Utilities for preprocessing step related tests."""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from template_pipelines.utils.optimization.io_utils import READ_CSV_NA_KWARGS


def get_sdm_io_tables() -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Return raw and result SDM tables."""
    data_path = Path("src/template_pipelines/tests/steps/optimization/test_data")
    raw_industries_df = pd.read_csv(
        data_path / "input_data" / "industries.csv", **READ_CSV_NA_KWARGS
    )
    raw_region_df = pd.read_csv(data_path / "input_data" / "regions.csv", **READ_CSV_NA_KWARGS)
    raw_stocks_df = pd.read_csv(data_path / "input_data" / "stocks.csv", **READ_CSV_NA_KWARGS)
    result_portfolio_df = pd.read_parquet(data_path / "input_data" / "portfolio_stocks.parquet")
    raw_general_inputs_df = pd.read_parquet(data_path / "input_data" / "general_inputs.parquet")
    return {
        "industries": raw_industries_df,
        "stocks": raw_stocks_df,
        "regions": raw_region_df,
        "general_inputs": raw_general_inputs_df,
    }, result_portfolio_df
