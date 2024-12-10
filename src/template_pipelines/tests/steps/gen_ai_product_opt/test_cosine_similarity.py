"""Test for cosine similarity."""

import logging
import random
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from langchain_core.documents import Document

from aif.pyrogai.steps.step import Step
from template_pipelines.steps.gen_ai_product_opt.cosine_similarity import CosineSimilarity


class MockEmbeddingModel:
    """A mock class for generating mock embeddings of a text."""

    def __call__(self, text):
        """Generate a mock embedding for a single text."""
        return self.embed_documents([text])[0]

    def embed_documents(self, texts):
        """Generate mock embeddings for multiple texts."""
        return [random.choices(np.arange(0, 1, 0.01), k=3) for t in texts]


class MockChatModel:
    """A mock class for generating mock chat responses."""

    def __call__(self, messages):
        """Generate a chat response based on input messages."""
        return self.get_chat_response(messages)

    def get_chat_response(self, messages):
        """Generate a chat response based on input messages."""
        message = messages.to_messages()[-1]
        response = message.dict()["content"].replace("\n", "").strip()
        return response


class MockBaseGenAIStep(Step):
    """A mock BaseGenAIStep for testing purposes."""

    def __init__(self):
        """Initialize a MockBaseGenAIStep instance."""
        super().__init__()

    def mock_super_init(self):
        """Mock initialization for setting up loggin, secrets and configuration."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging
        self.secrets = {}
        self.config = {
            "gen_ai_product_opt": {
                "distance_strategy": "COSINE",
                "similarity_threshold": 0.2,
                "num_results": 1,
            }
        }


@pytest.fixture(scope="function")
def fixture_cosine_similarity():
    """Fixture for the CosineSimilarity step."""
    with (
        patch(
            "template_pipelines.steps.gen_ai_product_opt.cosine_similarity.BaseGenAIStep.__init__",
            MockBaseGenAIStep.mock_super_init,
        ),
        patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"),
    ):
        cs = CosineSimilarity()
        yield cs


@pytest.fixture(scope="function")
def test_documents():
    """Fixture for providing test documents."""
    docs = ["text1", "text2"]
    yield [Document(page_content=doc, metadata=dict(page=0)) for doc in docs]


@pytest.fixture(scope="function")
def test_data():
    """Fixture for providing test data."""
    yield pd.DataFrame(
        {
            "product_id": ["1", "2"],
            "product_category": ["a", "b"],
            "title": ["title_a", "title_b"],
            "product_description": ["desc_1", "desc_2"],
            "keywords": ["text1,text2", "word1,word2"],
        }
    )


def test_format_docs(fixture_cosine_similarity, test_documents):
    """Test format_docs."""
    experected_doc_format = "text1,text2"
    res = fixture_cosine_similarity.format_docs(test_documents)

    assert isinstance(res, str)
    assert res == experected_doc_format


def test_create_documents(fixture_cosine_similarity, test_documents):
    """Test create_documents."""
    texts = "text1,text2"
    res = fixture_cosine_similarity.create_documents(texts, 0)

    assert isinstance(res, list)
    assert isinstance(res[0], Document)
    assert res == test_documents


def test_create_doc_array(fixture_cosine_similarity, test_data, test_documents):
    """Test create_doc_array."""
    test_row = test_data.iloc[[0], :]
    res = fixture_cosine_similarity.create_doc_array(test_row, "keywords")

    assert isinstance(res, np.ndarray)
    assert list(res) == test_documents


@patch("template_pipelines.steps.gen_ai_product_opt.cosine_similarity.pd.DataFrame.to_parquet")
@patch("template_pipelines.steps.gen_ai_product_opt.cosine_similarity.pd.read_parquet")
def test_cosine_similarity_run(
    mock_read_parquet, mock_to_parquet, fixture_cosine_similarity, test_data
):
    """Test the CosineSimilarity run method."""
    mock_read_parquet.return_value = test_data
    mock_to_parquet.return_value = True

    fixture_cosine_similarity.ioctx = MagicMock()
    fixture_cosine_similarity.genai_client = MagicMock()
    fixture_cosine_similarity.genai_client.embedding_model = MockEmbeddingModel()
    fixture_cosine_similarity.genai_client.chat_model = MockChatModel()
    fixture_cosine_similarity.run()

    expected_columns = [
        "product_id",
        "product_category",
        "title",
        "product_description",
        "keywords",
        "updated_keywords",
    ]

    assert list(test_data.columns) == expected_columns
    mock_read_parquet.assert_called_once_with(
        fixture_cosine_similarity.ioctx.get_fn("preprocessed.parquet")
    )
    mock_to_parquet.assert_called_once_with(
        fixture_cosine_similarity.ioctx.get_output_fn("cosine_similarity.parquet")
    )
