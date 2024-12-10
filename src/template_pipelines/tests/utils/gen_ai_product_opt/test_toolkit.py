"""Tests for toolkit."""

from datetime import datetime, timedelta
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts.chat import ChatPromptTemplate

from template_pipelines.utils.gen_ai_product_opt import toolkit


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


class MockToken:
    """A mock class for representing authentication tokens."""

    def __init__(self):
        """Initialize a MockToken instance with an expiration timestamp."""
        self.expires_on = (datetime.now() + timedelta(minutes=2)).timestamp()
        self.token = str(self.expires_on)


def test_create_chat_prompt():
    """Test create_chat_prompt."""
    test_system_prompt = "system prompt: {input}"
    test_user_prompt = "user prompt: {input}"
    res = toolkit.create_chat_prompt(test_system_prompt, test_user_prompt)

    assert isinstance(res, ChatPromptTemplate)
    assert len(res.messages) == 2


@patch("template_pipelines.utils.gen_ai_product_opt.toolkit.DefaultAzureCredential")
@patch("template_pipelines.utils.gen_ai_product_opt.toolkit.EnvironmentCredential")
@patch("template_pipelines.utils.gen_ai_product_opt.toolkit.ChainedTokenCredential")
def test_get_azure_token(mock_chained_credential, mock_env_credential, mock_default_credential):
    """Test get_azure_token."""
    mock_token = MagicMock()
    mock_token.token = "mock-token"
    mock_env_credential.return_value.get_token.return_value = mock_token
    mock_default_credential.return_value.get_token.return_value = mock_token
    mock_chained_credential.return_value.get_token.return_value = mock_token

    toolkit.get_azure_token("test")

    mock_chained_credential().get_token.assert_called_once_with("test")


class TestAzureOpenAIClient(TestCase):
    """Test cases for the AzureOpenAIClient class."""

    @patch("template_pipelines.utils.gen_ai_product_opt.toolkit.AzureOpenAIEmbeddings")
    @patch("template_pipelines.utils.gen_ai_product_opt.toolkit.AzureChatOpenAI")
    @patch("template_pipelines.utils.gen_ai_product_opt.toolkit.get_azure_token")
    def setUp(self, mock_azure_token, mock_azure_chat, mock_embedding):
        """Set up a test environment before each test method is executed."""
        mock_azure_token.return_value = MockToken()
        mock_azure_chat.return_value = MockChatModel()
        mock_embedding.return_value = MockEmbeddingModel()
        self.azure_openai_client = toolkit.AzureOpenAIClient(
            {
                "genai_proxy": "",
                "cognitive_services": "",
                "open_api_version": "",
                "headers": "",
                "token_refresh_thre": 10,
                "chat_engine": "",
                "temperature": "",
                "embedding_engine": "",
            }
        )

    def tearDown(self):
        """Remove the test environment after each test method is executed."""
        pass

    @patch("template_pipelines.utils.gen_ai_product_opt.toolkit.AzureOpenAIEmbeddings")
    @patch("template_pipelines.utils.gen_ai_product_opt.toolkit.AzureChatOpenAI")
    def test_initialize_genai_models(self, mock_azure_chat, mock_embedding):
        """Test the initialize_genai_models method."""
        mock_azure_chat.return_value = MockChatModel()
        mock_embedding.return_value = MockEmbeddingModel()
        self.azure_openai_client.initialize_genai_models()

        self.assertIsInstance(self.azure_openai_client.chat_model, MockChatModel)
        self.assertIsInstance(self.azure_openai_client.embedding_model, MockEmbeddingModel)

    def test_get_chat_response(self):
        """Test the get_chat_response method."""
        messages = {"system": "hello", "human": "world"}
        res = self.azure_openai_client.get_chat_response(messages)

        self.assertEqual(res, "hello world")

    def test_get_embedding(self):
        """Test the get_embedding method."""
        res = self.azure_openai_client.get_embedding(["hello world"])
        expected_res = [[1, 1, 1]]

        self.assertEqual(res, expected_res)

    @patch("template_pipelines.utils.gen_ai_product_opt.toolkit.AzureOpenAIEmbeddings")
    @patch("template_pipelines.utils.gen_ai_product_opt.toolkit.AzureChatOpenAI")
    @patch("template_pipelines.utils.gen_ai_product_opt.toolkit.get_azure_token")
    def test_get_embedding_with_token_refreshed(
        self, mock_azure_token, mock_azure_chat, mock_embedding
    ):
        """Test the get_embedding method with token refreshed."""
        mock_azure_token.return_value = MockToken()
        previous_token_expiration = self.azure_openai_client.token_expiration
        previous_token_str = self.azure_openai_client.token_str

        self.azure_openai_client.token_refresh_thre = 180
        self.azure_openai_client.get_embedding(["hello world"])

        new_token_expiration = self.azure_openai_client.token_expiration
        new_token_str = self.azure_openai_client.token_str

        self.assertGreater(new_token_expiration, previous_token_expiration)
        self.assertIsNot(new_token_str, previous_token_str)

    @patch("template_pipelines.utils.gen_ai_product_opt.toolkit.AzureOpenAIEmbeddings")
    @patch("template_pipelines.utils.gen_ai_product_opt.toolkit.AzureChatOpenAI")
    @patch("template_pipelines.utils.gen_ai_product_opt.toolkit.get_azure_token")
    def test_get_get_chat_response_with_token_refreshed(
        self, mock_azure_token, mock_azure_chat, mock_embedding
    ):
        """Test the get_chat_response method with token refreshed."""
        mock_azure_token.return_value = MockToken()
        previous_token_expiration = self.azure_openai_client.token_expiration
        previous_token_str = self.azure_openai_client.token_str

        self.azure_openai_client.token_refresh_thre = 180
        self.azure_openai_client.get_chat_response(messages={"system": "hello", "human": "world"})

        new_token_expiration = self.azure_openai_client.token_expiration
        new_token_str = self.azure_openai_client.token_str

        self.assertGreater(new_token_expiration, previous_token_expiration)
        self.assertIsNot(new_token_str, previous_token_str)


class TestVectorStoreRetrieverFilter(TestCase):
    """Test cases for the VectorStoreRetrieverFilter class."""

    def setUp(self):
        """Set up a test environment before each test method is executed."""
        self.doc_array = np.array(
            [
                Document(page_content="hello world", metadata={"page": 0}),
            ]
        )
        self.retriever = toolkit.VectoreStoreRetrieverFilter(
            vectorstore=FAISS.from_documents(
                self.doc_array,
                MockEmbeddingModel(),
                distance_strategy="COSINE",
            ),
            search_kwargs={
                "score_threshold": 0.2,
                "k": 5,
            },
        )

    def tearDown(self):
        """Remove the test environment after each test method is executed."""
        pass

    def test_get_relevant_documents2(self):
        """Test the _get_relevant_documents2 method."""
        res = self.retriever._get_relevant_documents2("hello world")
        res2 = self.retriever._get_relevant_documents2("hello world", _filter=dict(page=0))

        self.assertEqual(res, self.doc_array)
        self.assertEqual(res2, self.doc_array)

    def test_get_relevant_documents(self):
        """Test the _get_relevant_documents method."""
        res = self.retriever._get_relevant_documents({"query": "hello world"})
        res2 = self.retriever._get_relevant_documents(
            {"query": "hello world", "_filter": dict(page=0)}
        )

        self.assertEqual(res, self.doc_array)
        self.assertEqual(res2, self.doc_array)
