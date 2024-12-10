"""Unit tests for notification.py."""
from unittest.mock import MagicMock, patch

import pytest

from template_pipelines.steps.operations.notification import NotificationStep


@pytest.fixture(scope="function")
def fixture_notification():
    """Fixture for NotificationStep."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        notification = NotificationStep()
        notification.logger = MagicMock()
        notification.trigger_error = MagicMock()
        notification.mlflow = MagicMock()
        notification.mlflow_utils = MagicMock()

        yield notification


def test_notification_no_error_triggered(fixture_notification):
    """Test run() where RMSE < 1 and error should not be triggered."""
    fixture_notification.mlflow.get_run.return_value.data.metrics = {"RMSE": 0.9}
    fixture_notification.run()
    fixture_notification.logger.info.assert_any_call("Running Notification...")
    fixture_notification.logger.info.assert_any_call("Notification is done.")

    assert fixture_notification.logger.info.call_count == 2

    fixture_notification.trigger_error.assert_not_called()


def test_notification_error_triggered(fixture_notification):
    """Test run() where RMSE >= 1 and error should be triggered."""
    fixture_notification.mlflow.get_run.return_value.data.metrics = {"RMSE": 1.0}
    fixture_notification.run()
    fixture_notification.logger.info.assert_any_call("Running Notification...")
    fixture_notification.logger.info.assert_any_call("Notification is done.")

    assert fixture_notification.logger.info.call_count == 2

    fixture_notification.trigger_error.assert_called_once()
