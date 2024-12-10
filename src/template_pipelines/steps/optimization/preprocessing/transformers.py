"""Stocks transformations."""
from typing import Dict

import pandas as pd


class StocksTransformer:
    """Transformation logic for stocks tables."""

    kept_columns = ["name", "expected_return", "is_risky", "region"]

    def merge_input_tables(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge input datasets.

        Args:
            data (typing.Dict[str, pd.DataFrame]): raw datasets

        Returns:
            pd.DataFrame: joined tables
        """
        portfolio_stocks = data["stocks"].merge(data["regions"], on="region", how="left")
        portfolio_stocks = portfolio_stocks.merge(data["industries"], on="industry", how="left")
        return portfolio_stocks

    def filter_columns(self, portfolio_stocks: pd.DataFrame) -> pd.DataFrame:
        """Filter ony relevant columns."""
        return portfolio_stocks[self.kept_columns]

    def calc_expected_returns(self, portfolio_stocks: pd.DataFrame) -> pd.DataFrame:
        """Calculate expected returns column."""
        portfolio_stocks["expected_return"] = (
            portfolio_stocks["avg_return"] * portfolio_stocks["region_index"]
        )
        return portfolio_stocks
