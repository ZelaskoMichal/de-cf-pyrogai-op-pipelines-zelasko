"""Tests logging."""
import logging
from unittest.mock import Mock

import pytest

from template_pipelines.utils.operations.logging_utils import log_function_info


class Example:
    """Testing class."""

    def __init__(self, logger):
        """init."""
        self.logger = logger

    @log_function_info
    def successful_method(self):
        """Successful method."""
        return "success"

    @log_function_info
    def failing_method(self):
        """Failing method."""
        raise ValueError("intentional failure")


def test_successful_execution():
    """test_successful_execution."""
    logger = Mock(spec=logging.Logger)
    example = Example(logger)
    result = example.successful_method()

    assert result == "success"

    assert logger.info.call_count == 6
    calls = [call[0][0] for call in logger.info.call_args_list]
    assert "### Start Example.successful_method ###" in calls[1]
    assert "### Example.successful_method completed. Execution time:" in calls[-2]


def test_exception_raised():
    """test_exception_raised."""
    logger = Mock(spec=logging.Logger)
    example = Example(logger)

    with pytest.raises(ValueError):
        example.failing_method()

    assert logger.info.call_count == 6
    calls = [call[0][0] for call in logger.info.call_args_list]
    assert "### Start Example.failing_method ###" in calls[1]
    assert "### Example.failing_method failed. Execution time:" in calls[-2]
