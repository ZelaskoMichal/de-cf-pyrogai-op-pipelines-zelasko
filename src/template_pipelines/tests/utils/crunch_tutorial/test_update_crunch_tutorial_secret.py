"""Test update_crunch_tutorial_secret module."""
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import mock_open, patch

import pytest

import template_pipelines.utils.crunch_tutorial.update_crunch_acr_secret as update_crunch_acr_secret  # noqa: I250


# Mocking the Crunch class and its method
class MockCrunch:
    """Mock class for Crunch to simulate the behavior of the real Crunch class."""

    @staticmethod
    def from_app_credentials(url, user_id, secret):
        """Mock method for creating a Crunch instance from application credentials.

        Args:
            url (str): The URL for the Crunch service.
            user_id (str): The user ID for authentication.
            secret (str): The secret for authentication.

        Returns:
            MockCrunch: An instance of the mocked Crunch class.
        """
        return MockCrunch()

    def generate_registry_token(self):
        """Mock method to generate a registry token.

        Returns:
            dict: A dictionary containing a mock token with the current creation time.
        """
        return {"token": {"creation_time": datetime.now(timezone.utc).isoformat()}}


# Unit tests for load_secrets function
def test_load_secrets_success():
    """Test the load_secrets function for successful loading of secrets from a JSON file."""
    mock_data = '{"key": "value"}'
    with patch("builtins.open", mock_open(read_data=mock_data)):
        assert update_crunch_acr_secret.load_secrets("dummy_path") == json.loads(mock_data)


def test_load_secrets_file_not_found():
    """Test the load_secrets function to ensure it raises a FileNotFoundError when the file is not found."""
    with pytest.raises(FileNotFoundError):
        with patch("builtins.open", side_effect=FileNotFoundError):
            update_crunch_acr_secret.load_secrets("dummy_path")


def test_load_secrets_json_decode_error():
    """Test the load_secrets function to ensure it raises a JSONDecodeError when the JSON is invalid."""
    with pytest.raises(json.JSONDecodeError):
        with patch("builtins.open", mock_open(read_data="invalid_json")):
            update_crunch_acr_secret.load_secrets("dummy_path")


def test_save_secrets_io_error():
    """Test the save_secrets function to ensure it raises an IOError when file writing fails."""
    secrets = {"key": "value"}
    with patch("builtins.open", mock_open()) as mocked_file:
        mocked_file.side_effect = IOError
        with pytest.raises(IOError):
            update_crunch_acr_secret.save_secrets("dummy_path", secrets)


# Unit tests for validate_secret function
def test_validate_secret_valid():
    """Test the validate_secret function to ensure it correctly validates a valid CRUNCH-ACR secret."""
    now = datetime.now(timezone.utc)
    secrets = {"CRUNCH-ACR": {"token": {"creation_time": now.isoformat()}}}
    with patch(
        "template_pipelines.utils.crunch_tutorial.update_crunch_acr_secret.load_secrets",
        return_value=secrets,
    ):
        with patch("builtins.print") as mocked_print:
            update_crunch_acr_secret.validate_secret("dummy_path")
            mocked_print.assert_called_once_with("CRUNCH-ACR secret is valid.")


def test_validate_secret_invalid():
    """Test the validate_secret function to ensure it correctly identifies an invalid CRUNCH-ACR secret."""
    past_time = datetime.now(timezone.utc) - timedelta(hours=2)
    secrets = {"CRUNCH-ACR": {"token": {"creation_time": past_time.isoformat()}}}
    with patch(
        "template_pipelines.utils.crunch_tutorial.update_crunch_acr_secret.load_secrets",
        return_value=secrets,
    ):
        with patch("builtins.print") as mocked_print:
            update_crunch_acr_secret.validate_secret("dummy_path")
            mocked_print.assert_called_once_with(
                "CRUNCH-ACR secret is not valid, "
                "check the creation time and expiry of CRUNCH-ACR secret in your secrets.json file."
            )


# Unit tests for main function
@patch("template_pipelines.utils.crunch_tutorial.update_crunch_acr_secret.Crunch", new=MockCrunch)
def test_main():
    """Test the main function."""
    secrets = {
        "CRUNCH-APP-USER-NAME": "user",
        "CRUNCH-APP-USER-SECRET": "secret",
        "CRUNCH-URL": "http://example.com",
        "CRUNCH-ACR": {
            "token": {
                "creation_time": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
            }
        },
    }

    with patch(
        "template_pipelines.utils.crunch_tutorial.update_crunch_acr_secret.load_secrets",
        return_value=secrets,
    ):
        with patch(
            "template_pipelines.utils.crunch_tutorial.update_crunch_acr_secret.save_secrets"
        ) as mock_save_secrets:
            with patch("builtins.print"):
                update_crunch_acr_secret.main("dummy_path")

                # Verify save_secrets is called with updated secrets
                assert "CRUNCH-ACR" in secrets
                assert "token" in secrets["CRUNCH-ACR"]
                assert "creation_time" in secrets["CRUNCH-ACR"]["token"]
                mock_save_secrets.assert_called_once_with("dummy_path", secrets)
