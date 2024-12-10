"""Tests for base_genai_step."""

import time
import unittest
from datetime import datetime
from unittest.mock import patch

from langchain.chains import QAGenerationChain
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from aif.genai_utils import GenAIToken
from aif.pyrogai.const import ProviderConfigEnvironments
from aif.pyrogai.pipelines.local.environment import LocalPlatformProvider
from aif.pyrogai.secrets.secret import Secret
from template_pipelines.utils.doc_vectorization.base_genai_step import BaseGenAIStep


class MockBaseGenAIStep(BaseGenAIStep):
    """A mock sublass of BaseGenAIStep for testing purposes."""

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
            "scope": "doc-vectorization",
            "runtime_params": {},
            "environment": ProviderConfigEnvironments.LOCAL,
            "config": "config.json",
            "override": "",
            "logger_debug": True,
            "config_module": "template_pipelines.config",
            "pipeline_name": "doc_vectorization",
            "step_name": "step_doc_vectorization",
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
            **args, run_timestamp=datetime.now()
        )
        mock_azure_credential.return_value = MockDefaultAzureCredential()
        self.base_genai_step = MockBaseGenAIStep()

    def tearDown(self):
        """Remove the test environment after each test method is executed."""

    def test__init__(self):
        """Test the initialization of the BaseGenAIStep instance."""
        self.assertIsInstance(self.base_genai_step.genai_token, GenAIToken)
        self.assertIsInstance(self.base_genai_step.chat_model, AzureChatOpenAI)
        self.assertIsInstance(self.base_genai_step.embedding_model, AzureOpenAIEmbeddings)
        self.assertIsInstance(self.base_genai_step.qa_ge_chain, QAGenerationChain)
