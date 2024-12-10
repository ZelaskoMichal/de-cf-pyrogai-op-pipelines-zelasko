"""Standard Data Model for stock portfolio problem."""
import logging
from pathlib import Path
from typing import Dict, Union

import pandas as pd

from aif.pyrogai.ioutils.iocontext import IoContext
from template_pipelines.steps.optimization.preprocessing.transformers import StocksTransformer
from template_pipelines.utils.optimization.io_utils import (
    load_tables,
    load_values,
    save_tables,
    save_values,
)

logger = logging.getLogger(__name__)


class StocksPortfolioSDM:
    """Standard Data Model for Stock Portfolio optimization use case."""

    def __init__(self) -> None:
        """Initialize SDM."""
        self.sdm_data: Dict[str, Union[pd.DataFrame, float]] = {}
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.raw_data_info: Dict[
            str, str
        ] = {}  # path mapping, needed to log to mlflow on some platforms

    def create(self, data_path: str, ioctx: IoContext) -> Dict[str, float]:
        """Create the SDM from raw data, run all processing logic.

        Args:
            data_path (str): path within ioctx where data resides
            ioctx (IoContext): reference to ioctx object

        Returns:
            Dict[str, float]: metrics from creation logic, combined across all used tables
        """
        self._load_input_data(data_path, ioctx)

        # run all transformations

        # some processings can be performed directly within create() or as designated methods...
        raw_table_metrics = self._profile_tables(self.raw_data)

        # or inside a separate class for particular use case
        stocks_transformer = StocksTransformer()

        # SDM simplifies argument passing - if logic requires many tables pass full data dicts
        portfolio_stocks = stocks_transformer.merge_input_tables(self.raw_data)
        portfolio_stocks = stocks_transformer.calc_expected_returns(portfolio_stocks)
        portfolio_stocks = stocks_transformer.filter_columns(portfolio_stocks)

        general_inputs = self.raw_data["general_inputs"]
        penalty_max_number_risky = general_inputs[
            general_inputs["parameter"] == "penalty_max_number_risky"
        ].value.iloc[0]
        penalty_min_number_per_region = general_inputs[
            general_inputs["parameter"] == "penalty_min_number_per_region"
        ].value.iloc[0]

        # NOTE: the larger the code base the easier it is to read code split into classes/modules - transformers
        self.sdm_data = {
            "portfolio_stocks": portfolio_stocks,
            "penalty_max_number_risky": penalty_max_number_risky,
            "penalty_min_number_per_region": penalty_min_number_per_region,
        }
        return raw_table_metrics

    def _load_input_data(self, data_path: str, ioctx: IoContext):
        """Load input/raw data from iocontext.

        Args:
            data_path (str): folder in ioctx to read from
            ioctx (IoContext): ioctx reference
        """
        file_paths = ioctx.get_fns(f"{data_path}/*.parquet")
        self.raw_data, self.raw_data_info = load_tables(file_paths)

    def _profile_tables(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Profile tables and return per table metrics, prefixed with table name for batch metrics logging.

        Args:
            data (Dict[str, pd.DataFrame]): datasets to scan

        Returns:
            Dict[str, float]: metric:value mapping, with metric including table name prefix
        """
        metrics = {}
        for name, df in data.items():
            logger.debug(f"{name} datatypes: \n{df.dtypes}")
            table_metrics = self._get_table_statistics(df)

            # prefix metrics with table to avoid naming collisions
            table_metrics = {f"{name}_{metric}": value for metric, value in table_metrics.items()}
            metrics.update(table_metrics)

        return metrics

    def _get_table_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze input table, calculate base statistics.

        Args:
            table (pd.DataFrame): table object

        Returns:
            Dict[str, float]: metrics
        """
        metrics = {}

        # rows is one statistic we want for every table
        # if there are others, they can be added here
        metrics["rows"] = float(len(df))

        return metrics

    def load_stored_sdm(
        self,
        path: str,
        ioctx: IoContext,
        tables_file_pattern: str = "*.parquet",
        values_filename: str = "values.json",
    ) -> None:
        """Load SDM object from tables stored in iocontext.

        Args:
            path (str): directory path to where sdm resides in ioctx
            ioctx (IoContext): ioctx reference
            tables_file_pattern (str): filter for files to read within path
            values_filename (str): name of file where values are stored
        """
        # load data from ioctx - SDM  used in formulate_and_solve
        table_path_pattern = Path(path) / tables_file_pattern
        logger.info(f"Reading SDM tables from {table_path_pattern}")
        file_paths = ioctx.get_fns(table_path_pattern.as_posix())
        sdm_tables, _ = load_tables(file_paths)

        values_path = Path(path) / values_filename
        logger.info(f"Reading SDM values from {values_path}")
        file_paths = ioctx.get_fns(values_path.as_posix())
        sdm_values = load_values(file_paths)

        self.sdm_data = {**sdm_tables, **sdm_values}

    def save(self, path: str, ioctx: IoContext) -> None:
        """Persist SDM as individual tables in ioctx.

        Args:
            path (str): relative path to where sdm resides in ioctx
            ioctx (IoContext): ioctx module reference
        """
        path = ioctx.get_output_fn(path)
        # split between tables to save as parquet and values to save as json
        sdm_tables = {}
        sdm_values = {}
        for k, v in self.sdm_data.items():
            if isinstance(v, (pd.DataFrame, pd.Series)):
                sdm_tables[k] = v
            else:
                sdm_values[k] = v
        save_tables(sdm_tables, path)
        save_values(sdm_values, path)
