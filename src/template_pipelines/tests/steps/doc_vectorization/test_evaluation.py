"""Tests for doc evaluation."""

import json
import logging
from unittest.mock import MagicMock, mock_open, patch

import pytest

from aif.pyrogai.steps.step import Step
from template_pipelines.steps.doc_vectorization.evaluation import Evaluation


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
def fixture_evaluation():
    """Fixture for the Evaluation step."""
    with (
        patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"),
        patch(
            "template_pipelines.steps.doc_vectorization.evaluation.BaseGenAIStep.__init__",
            MockBaseGenAIStep.mock_super_init,
        ),
    ):
        evaluation = Evaluation()
        yield evaluation


@patch("template_pipelines.steps.doc_vectorization.evaluation.json.dump")
@patch("template_pipelines.steps.doc_vectorization.evaluation.NamedTemporaryFile")
@patch("template_pipelines.steps.doc_vectorization.evaluation.np.random.randint")
@patch(
    "template_pipelines.steps.doc_vectorization.evaluation.open",
    new_callable=mock_open,
    read_data=json.dumps(
        {
            "texts": ["text1", "text2"],
            "metadatas": [{"source": "file1.md"}, {"source": "file2.md"}],
            "ids": ["id1", "id2"],
        }
    ),
)
@patch("template_pipelines.steps.doc_vectorization.evaluation.FAISS.load_local")
def test_evaluation_run(
    mock_faiss_load,
    mock_file,
    mock_randint,
    mock_named_tempfile,
    mock_json_dump,
    fixture_evaluation,
):
    """Test the Evaluation run method."""
    mock_vectorstore = MagicMock()
    mock_faiss_load.return_value = mock_vectorstore
    mock_randint.return_value = [0, 1]

    fixture_evaluation.inputs = {
        "faiss_vector_db": "mock_faiss_vector_db",
        "aggs.json": "mock_aggs.json",
    }
    fixture_evaluation.outputs = {}

    fixture_evaluation.run()

    mock_faiss_load.assert_called_once_with(
        "mock_faiss_vector_db",
        fixture_evaluation.embedding_model,
        allow_dangerous_deserialization=True,
    )
    mock_file.assert_called_once_with("mock_aggs.json", "rb")

    for i in range(2):
        fixture_evaluation.qa_ge_chain.run.assert_any_call(f"text{i+1}")

    assert mock_vectorstore.similarity_search.call_count == 2
    mock_named_tempfile.assert_called_once()
    mock_json_dump.assert_called_once()
    assert "evaluation.json" in fixture_evaluation.outputs
