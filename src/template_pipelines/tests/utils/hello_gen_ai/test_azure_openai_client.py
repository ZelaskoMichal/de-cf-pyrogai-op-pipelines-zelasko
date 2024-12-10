"""Tests for azure_openai_client."""

from unittest import TestCase
from unittest.mock import MagicMock, patch

from template_pipelines.utils.hello_gen_ai import azure_openai_client


class ValueWithAttribute:
    """A class that represents a value with attribute access."""

    def __init__(self, value):
        """Initialize ValueWithAttribute with a value."""
        self.value = value

    def __getattr__(self, attr):
        """Retrieve a value associated with its object."""
        return self.value


class MockEmbeddingModel:
    """A mock class for generating mock embeddings of a text."""

    def __call__(self, text):
        """Generate a mock embedding for a single text."""
        return self.embed_documents([text])[0]

    def embed_documents(self, texts):
        """Generate mock embeddings for multiple texts."""
        return [[1, 1, 1] for t in texts]


class MockChatModel:
    """A mock class for generating mock chat responses."""

    def __call__(self, messages):
        """Generate a chat response based on input messages."""
        return self.get_chat_response(messages)

    def get_chat_response(self, messages):
        """Generate a chat response based on input messages."""
        message_list = []
        for message in messages:
            message_list.append(message.content)
        return ValueWithAttribute(" ".join(message_list))


class TestAzureOpenAIClient(TestCase):
    """Test cases for the AzureOpenAIClient class."""

    @patch("template_pipelines.utils.hello_gen_ai.azure_openai_client.EnvironmentCredential")
    @patch("template_pipelines.utils.hello_gen_ai.azure_openai_client.DefaultAzureCredential")
    @patch("template_pipelines.utils.hello_gen_ai.azure_openai_client.ChainedTokenCredential")
    def setUp(self, mock_chained_credential, mock_default_credential, mock_env_credential):
        """Set up a test environment before each test method is executed."""
        mock_token = MagicMock()
        mock_token.token = "mock-token"
        mock_env_credential.return_value.get_token.return_value = mock_token
        mock_default_credential.return_value.get_token.return_value = mock_token
        mock_chained_credential.return_value.get_token.return_value = mock_token
        self.azure_openai_client = azure_openai_client.AzureOpenAIClient(
            {
                "genai_proxy": "",
                "cognitive_services": "",
                "open_api_version": "",
                "headers": "",
                "temperature": "",
            },
            "chat_engine",
            "embedding_engine",
        )

    def tearDown(self):
        """Remove the test environment after each test method is executed."""
        pass

    @patch("template_pipelines.utils.hello_gen_ai.azure_openai_client.AzureChatOpenAI")
    def test_get_chat_response(self, mock_azure_chat):
        """Test the get_chat_response method."""
        mock_azure_chat.return_value = MockChatModel()
        messages = {"system": "hello", "human": "world"}
        res = self.azure_openai_client.get_chat_response(messages)

        mock_azure_chat.assert_called_once()
        self.assertEqual(res, "hello world")

    @patch("template_pipelines.utils.hello_gen_ai.azure_openai_client.AzureOpenAIEmbeddings")
    def test_get_embedding(self, mock_embedding):
        """Test the get_embedding method."""
        mock_embedding.return_value = MockEmbeddingModel()
        res = self.azure_openai_client.get_embedding(["hello world"])
        expected_res = [[1, 1, 1]]

        mock_embedding.assert_called_once()
        self.assertEqual(res, expected_res)
