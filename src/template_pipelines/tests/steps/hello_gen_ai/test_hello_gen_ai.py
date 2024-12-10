"""Test for hello genai."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from aif.pyrogai.steps.step import Step
from template_pipelines.steps.hello_gen_ai.hello_gen_ai import HelloGenAI


class MockBaseGenAIStep(Step):
    """A mock BaseGenAIStep for testing purposes."""

    def __init__(self):
        """Initialize a MockBaseGenAIStep instance."""
        super().__init__()

    def mock_super_init(self):
        """Mock initialization for setting up loggin, secrets and configuration."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging
        self.config = {"gen_ai": {"temperature": 0.2}}


@pytest.fixture(scope="function")
def fixture_hello_genai():
    """Fixture for the HelloGenAI step."""
    with (
        patch(
            "template_pipelines.steps.hello_gen_ai.hello_gen_ai.BaseGenAIStep.__init__",
            MockBaseGenAIStep.mock_super_init,
        ),
        patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"),
    ):
        hg = HelloGenAI()
        yield hg


def test_hello_genai_run(fixture_hello_genai):
    """Test the CosineSimilarity run method."""
    messages = {"system_message": "hello", "user_message": "world"}
    chat_response = "hello world"

    fixture_hello_genai.runtime_parameters = messages
    fixture_hello_genai.genai_client = MagicMock()
    fixture_hello_genai.genai_client.get_chat_response.return_value = chat_response
    fixture_hello_genai.genai_client.get_embedding.return_value = [1, 1, 1]
    fixture_hello_genai.run()

    fixture_hello_genai.genai_client.get_chat_response.assert_called_once_with(
        {"system": messages["system_message"], "human": messages["user_message"]}, temperature=0.2
    )
    fixture_hello_genai.genai_client.get_embedding.assert_called_once_with(chat_response.split(" "))
