"""Unit tests for LoggingStep class."""
import time  # noqa
from unittest.mock import MagicMock, patch

import pytest

from template_pipelines.steps.operations.logging import LoggingStep


@pytest.fixture(scope="function")
def logging_step():
    """Fixture for logging step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        ls = LoggingStep()
        ls.logger = MagicMock()
        yield ls


def test_run(logging_step):
    """Test run method of LoggingStep."""
    logging_step.run()

    # Check if logger methods are called appropriately
    assert logging_step.logger.debug.called_with("debug logging")
    assert logging_step.logger.info.called_with("info logging")
    assert logging_step.logger.warning.called_with("warning logging")
    assert logging_step.logger.error.called_with("error logging")
    assert logging_step.logger.critical.called_with("critical logging")
    assert logging_step.logger.fatal.called_with("fatal logging")

    # Check if 'some_random_func' also called its logger methods
    logging_step.some_random_func()
    assert logging_step.logger.info.called_with("some random info")
    assert logging_step.logger.warning.called_with("some random warning")


def test_some_random_func(logging_step):
    """Test some_random_func method of LoggingStep."""
    with patch("time.sleep", return_value=None):  # Avoid actual sleeping in the test
        logging_step.some_random_func()

    # Check if logger methods are called appropriately
    logging_step.logger.info.assert_any_call("some random info")
    logging_step.logger.info.assert_any_call("some random warning")
