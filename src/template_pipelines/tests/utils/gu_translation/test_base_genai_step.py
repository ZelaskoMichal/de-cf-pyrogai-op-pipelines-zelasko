"""Tests for base_genai_step."""

import os
import time
import unittest
from datetime import datetime
from unittest.mock import patch

import httpx

from aif.genai_utils import GenAIToken
from aif.pyrogai.const import ProviderConfigEnvironments
from aif.pyrogai.pipelines.local.environment import LocalPlatformProvider
from aif.pyrogai.secrets.secret import Secret
from template_pipelines.utils.gu_translation.base_genai_step import BaseGenAIStep


class MockBaseGenAIStep(BaseGenAIStep):
    """A mock sublass of BaseGenAIStep for testing purposes."""

    def __init__(self):
        """Initialize a MockBaseGenAIStep instance."""
        super().__init__()
        expected_url = "https://api.genai.com"
        self.config = {
            "gu_translation": {
                "genai_url": expected_url,
                "service_endpoint": "/service_endpoint",
                "headers": {"Content-Type": "application/json"},
            }
        }

    def run(self):
        """A mock run method."""
        pass


class MockSecret(Secret):
    """A mock subclass of Secret based on a dictionary."""

    def __init__(self, secrets):
        """Initialize a MockSecret instance  with a dictionary of secrets."""
        self._secrets = secrets
        self.environment = ProviderConfigEnvironments.LOCAL
        self._pooled_secrets = []
        self._pooled_secrets_cache = {}

    def _get_from_backend(self, key):
        """Retrieve a secret value from the dictionary of secrets."""
        try:
            return self._secrets[key]
        except KeyError:
            return None


class ValueWithAttribute:
    """A class that represents an object with dynamic attribute access."""

    def __init__(self, *args):
        """Initialize a ValueWithAttribute instance with a mock token and expiration time."""
        self.token = "mock_token"
        self.expires_on = time.time() + 9999

    def __getattr__(self, attr):
        """Handle access to undefined attributes by returning a tuple of token and expiration time."""
        return (self.token, self.expires_on)


class MockDefaultAzureCredential:
    """A mock class simulating Azure's DefaultAzureCredential for testing purposes."""

    def get_token(self, *args):
        """Simulate the retrieval of an authentication token."""
        return ValueWithAttribute(*args)


class TestBaseGenAIStep(unittest.TestCase):
    """Test cases for the BaseGenAIStep class."""

    @patch("aif.genai_utils.azure.genai_token.DefaultAzureCredential", autospec=True)
    @patch("aif.pyrogai.pipelines.local.environment.LocalSecret")
    @patch("aif.pyrogai.pipelines.components.environment.PlatformProvider")
    def setUp(self, mock_platform_provider, mock_local_secret, mock_azure_credential):
        """Set up a test environment before each test method is executed."""
        args = {
            "scope": "gu-translation",
            "runtime_params": {},
            "environment": ProviderConfigEnvironments.LOCAL,
            "config": "config.json",
            "override": "",
            "logger_debug": True,
            "config_module": "template_pipelines.config",
            "pipeline_name": "gu_translation",
            "step_name": "step_translation",
            "experiment_name": None,
            "github_deployment_url": None,
            "github_commit_status_url": None,
            "provider_name": "Local Provider",
        }
        self.secrets = {
            "tenant-id": "AZURE_TENANT_ID",
            "AML-APP-SP-ID": "AZURE_CLIENT_ID",
            "AML-APP-SP-SECRET": "AZURE_CLIENT_SECRET",
        }

        mock_local_secret.return_value = MockSecret(self.secrets)
        mock_platform_provider.return_value = LocalPlatformProvider.set_instance(
            **args,
            run_timestamp=datetime.now(),
        )
        mock_azure_credential.return_value = MockDefaultAzureCredential()

        self.base_genai_step = MockBaseGenAIStep()

    def tearDown(self):
        """Remove the test environment after each test method is executed."""

    def test__init__(self):
        """Test the initialization of the BaseGenAIStep instance."""
        for value in self.secrets.values():
            self.assertEqual(os.environ[value], value)
        self.assertIsInstance(self.base_genai_step.genai_token, GenAIToken)

    @patch("template_pipelines.utils.gu_translation.base_genai_step.httpx.Client.post")
    def test_translate(self, mock_post):
        """Test the successful translation request with the GenAI platform."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"translation": "I love Pho!"}

        text = "Uwielbiam Pho!"
        original = "PL"
        target = "EN"
        response = self.base_genai_step.translate(text, original, target)

        expected_url = "https://api.genai.com/service_endpoint"
        expected_headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer mock_token",
        }
        expected_body = {
            "text_to_translate": text,
            "source_language": original,
            "target_language": target,
            "expected_tone": "official",
            "list_of_terms": {"GU": "Generalized Utility"},
        }

        mock_post.assert_called_once_with(
            expected_url, headers=expected_headers, json=expected_body, timeout=60
        )
        self.assertEqual(response, "I love Pho!")

    @patch("template_pipelines.utils.gu_translation.base_genai_step.httpx.Client.post")
    def test_translate_http_error(self, mock_post):
        """Test handling of HTTP errors during sending a translation request to the GenAI Platform."""
        mock_post.side_effect = httpx.HTTPStatusError("Bad request", request=None, response=None)
        text = "Uwielbiam Pho!"
        original = "PL"
        target = "EN"

        with unittest.TestCase().assertLogs(level="ERROR") as log:
            response = self.base_genai_step.translate(text, original, target)
        msg = log.records[-1].msg

        self.assertEqual(
            msg, f"HTTP error for the translation from {original} to {target}: Bad request"
        )
        self.assertEqual(response, "Error during translation")

    @patch("template_pipelines.utils.gu_translation.base_genai_step.httpx.Client.post")
    def test_translate_generic_error(self, mock_post):
        """Test handling of generic errors during sending a translation request to the GenAI Platform."""
        mock_post.side_effect = Exception("Unexpected error")
        text = "Uwielbiam Pho!"
        original = "PL"
        target = "EN"

        with unittest.TestCase().assertLogs(level="ERROR") as log:
            response = self.base_genai_step.translate(text, original, target)
        msg = log.records[-1].msg

        self.assertEqual(
            msg, f"Error for the translation from {original} to {target}: Unexpected error"
        )
        self.assertEqual(response, "Error during translation")
