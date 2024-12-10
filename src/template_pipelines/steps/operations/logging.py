"""Logging step class."""

import time

from aif.pyrogai.steps.step import Step  # noqa
from template_pipelines.utils.operations.logging_utils import log_function_info


class LoggingStep(Step):
    """Logging step."""

    @log_function_info
    def some_random_func(self):
        """Random func."""
        self.logger.info("some random info")
        time.sleep(1.01)
        self.logger.info("some random warning")

    @log_function_info  # check decorator info in template_pipelines.utils.operations.logging_utils
    def run(self):
        """Run logging step."""
        self.logger.debug("debug logging")  # to see debug you need to run pipeline with '--debug'
        self.logger.info("info logging")
        self.logger.warning("warning logging")
        self.logger.error("error logging")
        self.logger.critical("critical logging")
        self.logger.fatal("fatal logging")

        self.some_random_func()
