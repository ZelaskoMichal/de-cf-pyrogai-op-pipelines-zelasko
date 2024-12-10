"""Test for content generation."""

import logging
from tempfile import NamedTemporaryFile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aif.pyrogai.steps.step import Step
from template_pipelines.steps.gen_ai_product_opt.content_generation import ContentGeneration


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
def fixture_content_generation():
    """Fixture for the ContentGeneration step."""
    with (
        patch(
            "template_pipelines.steps.gen_ai_product_opt.content_generation.BaseGenAIStep.__init__",
            MockBaseGenAIStep.mock_super_init,
        ),
        patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"),
    ):
        cg = ContentGeneration()
        yield cg


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
            "updated_keywords": ["updatedtest1,updatedtext2", "updatedword1,updatedword2"],
        }
    )


@patch("template_pipelines.steps.gen_ai_product_opt.content_generation.pd.DataFrame.to_parquet")
@patch("template_pipelines.steps.gen_ai_product_opt.content_generation.NamedTemporaryFile")
@patch("template_pipelines.steps.gen_ai_product_opt.content_generation.pd.read_parquet")
def test_content_generation_run(
    mock_read_parquet,
    mock_named_temporary_file,
    mock_to_parquet,
    fixture_content_generation,
    test_data,
):
    """Test the ContentGeneration run method."""
    mock_file = NamedTemporaryFile()
    mock_read_parquet.return_value = test_data
    mock_named_temporary_file.return_value = mock_file
    mock_to_parquet.return_value = True

    fixture_content_generation.ioctx = MagicMock()
    fixture_content_generation.genai_client = MagicMock()
    fixture_content_generation.genai_client.chat_model = MockChatModel()
    fixture_content_generation.outputs = {}
    fixture_content_generation.run()

    expected_columns = [
        "product_id",
        "product_category",
        "title",
        "product_description",
        "keywords",
        "updated_keywords",
        "optimized_title",
        "optimized_description",
    ]

    assert list(test_data.columns) == expected_columns
    mock_read_parquet.assert_called_once_with(
        fixture_content_generation.ioctx.get_fn("updated_keywords.parquet")
    )
    mock_to_parquet.assert_called_once_with(mock_file.name)
