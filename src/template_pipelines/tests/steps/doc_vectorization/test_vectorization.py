"""Tests for doc vectorization."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from aif.pyrogai.steps.step import Step
from template_pipelines.steps.doc_vectorization.vectorization import Vectorization


class MockBaseGenAIStep(Step):
    """A mock BaseGenAIStep for testing purposes."""

    def __init__(self):
        """Initialize a MockBaseGenAIStep instance."""
        super().__init__()

    def mock_super_init(self):
        """Mock initialization for setting up logging, secrets and configuration."""
        self.logger = logging
        self.qa_ge_chain = MagicMock()
        self.embedding_model = MagicMock()
        self.secrets = {}


@pytest.fixture(scope="function")
def fixture_vectorization():
    """Fixture for the Vectorization step."""
    with (
        patch(
            "template_pipelines.steps.doc_vectorization.vectorization.BaseGenAIStep.__init__",
            MockBaseGenAIStep.mock_super_init,
        ),
        patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"),
    ):
        vectorization = Vectorization()
        yield vectorization


@patch("template_pipelines.steps.doc_vectorization.vectorization.InMemoryDocstore")
@patch("template_pipelines.steps.doc_vectorization.vectorization.faiss.IndexFlatL2")
@patch("template_pipelines.steps.doc_vectorization.vectorization.FAISS")
@patch("template_pipelines.steps.doc_vectorization.vectorization.QuestionGenerationEmbedding")
def test__init__(
    mock_qa_embedding, mock_faiss, mock_faiss_index, mock_docstore, fixture_vectorization
):
    """Test the initialization of the Vectorization class."""
    fixture_vectorization.__init__()

    mock_qa_embedding.assert_called_once_with(
        question_generation_chain=fixture_vectorization.qa_ge_chain,
        embedding_model=fixture_vectorization.embedding_model,
    )

    mock_faiss.assert_called_once_with(
        embedding_function=mock_qa_embedding.return_value,
        index=mock_faiss_index.return_value,
        docstore=mock_docstore.return_value,
        index_to_docstore_id={},
    )

    assert fixture_vectorization.vectorstore == mock_faiss.return_value


@pytest.mark.asyncio
async def test_add_batch_async(fixture_vectorization):
    """Test add_batch_async."""
    mock_batch = {
        "texts": ["text1", "text2"],
        "metadatas": [{"source": "file1.md"}, {"source": "file2.md"}],
        "ids": ["id1", "id2"],
    }
    fixture_vectorization.vectorstore.aadd_texts = AsyncMock()

    await fixture_vectorization.add_batch_async(mock_batch)

    fixture_vectorization.vectorstore.aadd_texts.assert_awaited_once_with(**mock_batch)


@pytest.mark.asyncio
async def test_process_texts_in_batches_async(fixture_vectorization):
    """Test the process_texts_in_batches_async."""
    text_details = {
        "texts": ["text1", "text2", "text3"],
        "metadatas": [{"source": "file1.md"}, {"source": "file2.md"}, {"source": "file3.md"}],
        "ids": ["id1", "id2", "id3"],
    }
    batch_size = 2
    fixture_vectorization.add_batch_async = AsyncMock()

    await fixture_vectorization.process_texts_in_batches_async(text_details, batch_size)

    expected_calls = [
        {
            "texts": ["text1", "text2"],
            "metadatas": [{"source": "file1.md"}, {"source": "file2.md"}],
            "ids": ["id1", "id2"],
        },
        {"texts": ["text3"], "metadatas": [{"source": "file3.md"}], "ids": ["id3"]},
    ]

    fixture_vectorization.add_batch_async.call_count == batch_size
    fixture_vectorization.add_batch_async.assert_any_await(expected_calls[0])
    fixture_vectorization.add_batch_async.assert_any_await(expected_calls[1])


@patch("template_pipelines.steps.doc_vectorization.vectorization.TemporaryDirectory")
@patch("template_pipelines.steps.doc_vectorization.vectorization.asyncio.run")
@patch.object(Vectorization, "process_texts_in_batches_async", new_callable=AsyncMock)
@patch(
    "template_pipelines.steps.doc_vectorization.vectorization.open",
    new_callable=mock_open,
    read_data=json.dumps(
        {
            "texts": ["text1", "text2"],
            "metadatas": [{"source": "file1.md"}, {"source": "file2.md"}],
            "ids": ["id1", "id2"],
        }
    ),
)
def test_vectorization_run(
    mock_file,
    mock_process_texts_in_batches_async,
    mock_asyncio_run,
    mock_temp_dir,
    fixture_vectorization,
):
    """Test the Vectorization run method."""
    fixture_vectorization.inputs = {"aggs.json": "mock_aggs.json"}
    fixture_vectorization.vectorstore = MagicMock()
    fixture_vectorization.outputs = {"faiss_vector_db": "mock_faiss_vector_db"}
    mock_temp_dir.return_value.__enter__.return_value = "/mock/tempdir"
    fixture_vectorization.run()

    mock_file.assert_called_once_with("mock_aggs.json", "rb")
    mock_file().read.assert_called_once()
    mock_process_texts_in_batches_async.assert_called_once_with(
        {
            "texts": ["text1", "text2"],
            "metadatas": [{"source": "file1.md"}, {"source": "file2.md"}],
            "ids": ["id1", "id2"],
        },
        batch_size=10,
    )
    mock_asyncio_run.assert_called_once()
    mock_temp_dir.assert_called_once()
    fixture_vectorization.vectorstore.save_local.assert_called_once_with("/mock/tempdir")
    assert fixture_vectorization.outputs["faiss_vector_db"] == "/mock/tempdir"
