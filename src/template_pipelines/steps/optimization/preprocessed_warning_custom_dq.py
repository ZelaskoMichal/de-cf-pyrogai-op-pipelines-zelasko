"""Step that checks sdm's table with custom validation."""
import logging
from tempfile import NamedTemporaryFile

from aif.pyrogai.steps.step import Step
from template_pipelines.steps.optimization.preprocessing.sdm import StocksPortfolioSDM
from template_pipelines.utils.optimization.logger_utils import (
    config_logger_for_file_handler,
    get_dq_warnings_logger,
)

dq_warnings_logger = get_dq_warnings_logger()


class PreprocessedWarningCustomDq(Step):
    """PreprocessedWarningCustomDq Step.

    If checks fail, step logs warnings. The warnings are accessible in same way as other pyrogai's
    logs and also in ioslot file - called dq_warnings.log. Notice that if there
    are no failed checks, dq_warnings.log is empty.
    """

    def run(self) -> None:
        """Config logger, validate data, save logs to ioslots and remove file handler in the end."""
        with NamedTemporaryFile() as f:
            file_handler = config_logger_for_file_handler(
                dq_warnings_logger,
                self.config["dq_warnings_logger_format"],
                logging.WARNING,
                f.name,
            )
            self.validate()
            self.outputs["dq_warnings.log"] = f.name  # Save file logs to ioslots.
            dq_warnings_logger.removeHandler(file_handler)

    def validate(self):
        """Validate data from SDM. Log warnings."""
        sdm = StocksPortfolioSDM()
        sdm.load_stored_sdm(path=self.config["sdm_tmp_dir"], ioctx=self.ioctx)
        portfolio_stocks = sdm.sdm_data["portfolio_stocks"]

        # No risky stocks
        no_risky_stock = portfolio_stocks["is_risky"].eq(0).all()
        if no_risky_stock:
            dq_warnings_logger.warning("There is no risky stock.")

        # Only 1 region
        is_only_one_region = portfolio_stocks["region"].drop_duplicates().shape[0] == 1
        if is_only_one_region:
            dq_warnings_logger.warning("There is only 1 region.")
