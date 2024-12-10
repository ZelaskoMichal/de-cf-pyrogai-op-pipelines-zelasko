"""Step that checks solutions with custom validation."""
import logging
from tempfile import NamedTemporaryFile

import pandas as pd

from aif.pyrogai.steps.step import Step
from template_pipelines.steps.optimization.preprocessing.sdm import StocksPortfolioSDM
from template_pipelines.utils.optimization.io_utils import load_tables
from template_pipelines.utils.optimization.logger_utils import (
    config_logger_for_file_handler,
    get_dq_warnings_logger,
)

dq_warnings_logger = get_dq_warnings_logger()


class SolutionCustomDq(Step):
    """SolutionCustomDq Step."""

    def run(self) -> None:
        """Apply checks to the solution tables.

        Checks:
            - tables have the same number of rows as the portfolio stocks table (SDM)
            - no more than 2 stocks have a greater than 15% share of portfolio
        """
        sdm = StocksPortfolioSDM()
        sdm.load_stored_sdm(path=self.config["sdm_tmp_dir"], ioctx=self.ioctx)
        expected_len = len(sdm.sdm_data["portfolio_stocks"])
        solution = load_tables(self.ioctx.get_fns(f"{self.config['solution_tmp_dir']}/*.parquet"))
        variables_to_check = ["buy", "fraction"]
        for variable in variables_to_check:
            variable_df = solution[0][variable]
            if len(variable_df) != expected_len:
                msg = (
                    f"{variable} solution doesn't have the same number of rows as portfolio_stocks"
                )
                raise ValueError(msg)

        # logs to warnings file if some stocks are bought in large quantities
        with NamedTemporaryFile() as f:
            file_handler = config_logger_for_file_handler(
                dq_warnings_logger,
                self.config["dq_warnings_logger_format"],
                logging.WARNING,
                f.name,
            )
            self._check_specific_stocks(solution[0]["fraction"])
            self.outputs["solution_dq_warnings.log"] = f.name  # Save file logs to ioslots.
            dq_warnings_logger.removeHandler(file_handler)

    def _check_specific_stocks(self, fraction_stocks: pd.DataFrame):
        """Checks if more than 2 stocks are bought to more than 15% of the portfolio."""
        large_fractions = fraction_stocks[fraction_stocks.fraction > 0.15]
        if len(large_fractions) > 2:
            dq_warnings_logger.warning(
                f"Some stocks will make more than 15% of your portfolio: {list(large_fractions.name)}"
            )
