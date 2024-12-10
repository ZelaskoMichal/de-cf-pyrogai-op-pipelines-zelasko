"""Tests for base_genai_step."""

from datetime import datetime
from unittest.mock import patch

from aif.pyrogai.const import ProviderConfigEnvironments
from aif.pyrogai.pipelines.local.environment import LocalPlatformProvider
from aif.pyrogai.secrets.secret import Secret
from template_pipelines.utils.gen_ai_product_opt.base_genai_step import BaseGenAIStep


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


class MockAzureOpenAIClient:
    """A mock AzureOpenAIClient class for testing purposes."""

    pass


@patch("template_pipelines.utils.gen_ai_product_opt.base_genai_step.AzureOpenAIClient")
@patch("aif.pyrogai.pipelines.local.environment.LocalSecret")
@patch("aif.pyrogai.pipelines.components.environment.PlatformProvider")
def test_base_genai_step(platform_provider, mock_local_secret, mock_azure_openai):
    """Test base_genai_step."""
    mock_local_secret.return_value = MockSecret(
        {"tenant-id": "123", "AML-APP-SP-ID": "TPT", "AML-APP-SP-SECRET": "TPT123"}
    )

    args = {
        "scope": "gen-ai-product-opt",
        "runtime_params": {},
        "environment": ProviderConfigEnvironments.LOCAL,
        "config": "config.json",
        "override": "",
        "logger_debug": True,
        "config_module": "template_pipelines.config",
        "pipeline_name": "gen_ai_product_opt",
        "step_name": "step_gen_ai",
        "experiment_name": None,
        "github_deployment_url": None,
        "github_commit_status_url": None,
        "provider_name": "Local Provider",
    }
    platform_provider.return_value = LocalPlatformProvider.set_instance(
        **args,
        run_timestamp=datetime.now(),
    )
    mock_azure_openai.return_value = MockAzureOpenAIClient()
    base_genai_step = MockBaseGenAIStep()

    mock_azure_openai.assert_called_once()
    assert isinstance(base_genai_step.genai_client, MockAzureOpenAIClient)
