"""Step that checks sdm's table with custom validation.

If any validations fail, pipeline will be stopped.
"""
import math

from aif.pyrogai.steps.step import Step
from template_pipelines.steps.optimization.preprocessing.sdm import StocksPortfolioSDM
from template_pipelines.utils.optimization.io_utils import cast_numeric_runtime_parameters


class PreprocessedCriticalCustomDq(Step):
    """PreprocessedCriticalCustomDq Step. It can raise error and stop entire pipeline."""

    runtime_parameters: dict

    def pre_run(self) -> None:
        """Cast runtime_parameters before running."""
        self.runtime_parameters = cast_numeric_runtime_parameters(self.runtime_parameters)

    def run(self) -> None:
        """Validate data. Raise ValueError in case of validation failures."""
        errors = []

        sdm = StocksPortfolioSDM()
        sdm.load_stored_sdm(path=self.config["sdm_tmp_dir"], ioctx=self.ioctx)
        portfolio_stocks = sdm.sdm_data["portfolio_stocks"]

        # Too few rows to meet max fraction constraints
        # e.g. if there are 5 rows we must assign at least one row more than 20%
        # but the maximum fraction per stock is 10%
        if len(portfolio_stocks) < math.ceil(
            1
            / min(
                self.runtime_parameters["max_ratio_per_stock"],
                self.runtime_parameters["max_risky_stocks_ratio"],
            )
        ):
            errors.append("Too few rows to meet max fraction constraints")

        # Too few rows in each region: for each region
        counted_table = portfolio_stocks.groupby("region").count()
        regions_to_log = []
        for region_name, row in counted_table.iterrows():
            if row["name"] < self.runtime_parameters["min_stocks_per_region"]:
                regions_to_log.append(region_name)
        if regions_to_log:
            errors.append(f"Too few rows for region: {', '.join(regions_to_log)}")

        if errors:
            error_msg = "; ".join(errors)
            raise ValueError(error_msg)
