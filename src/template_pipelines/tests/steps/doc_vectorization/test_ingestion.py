"""Tests for doc ingestion."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_pipelines.steps.doc_vectorization.ingestion import Ingestion


@pytest.fixture(scope="function")
def fixture_ingestion():
    """Fixture for the Ingestion step."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        ingestion = Ingestion()
        yield ingestion


@pytest.mark.asyncio
@patch("template_pipelines.steps.doc_vectorization.ingestion.UnstructuredMarkdownLoader")
async def test_load_document_async(mock_loader, fixture_ingestion):
    """Test load_document_async."""
    mock_loader_instance = mock_loader.return_value
    mock_loader_instance.load.return_value = "document content"

    path = "test.md"
    result = await fixture_ingestion.load_document_async(path)

    mock_loader.assert_called_once_with(path)
    mock_loader_instance.load.assert_called_once()
    assert result == "document content"


@pytest.mark.asyncio
async def test_run_tasks(fixture_ingestion):
    """Test run_tasks."""
    mock_task1 = AsyncMock(return_value=["document 1"])
    mock_task2 = AsyncMock(return_value=["document 2", "document 3"])

    tasks = [mock_task1(), mock_task2()]
    result = [doc async for doc in fixture_ingestion.run_tasks(tasks)]
    assert result == ["document 1", "document 2", "document 3"]


@pytest.mark.asyncio
@patch("template_pipelines.steps.doc_vectorization.ingestion.RecursiveCharacterTextSplitter")
async def test_load(mock_text_splitter, fixture_ingestion):
    """Test load."""
    mock_doc1 = MagicMock(page_content="Content 1", metadata={"source": "doc1"})
    mock_doc2 = MagicMock(page_content="Content 2", metadata={"source": "doc2"})
    mock_splitter_instance = mock_text_splitter.return_value
    mock_splitter_instance.split_text.side_effect = [["Chunk 1.1", "Chunk 1.2"], ["Chunk 2.1"]]

    async def mock_run_tasks(tasks):
        yield mock_doc1
        yield mock_doc2

    fixture_ingestion.run_tasks = mock_run_tasks

    tasks = ["task1", "task2"]
    result = await fixture_ingestion.load(tasks)
    expected = {
        "metadatas": [{"source": "doc1/0/pl"}, {"source": "doc1/1/pl"}, {"source": "doc2/0/pl"}],
        "texts": ["Chunk 1.1", "Chunk 1.2", "Chunk 2.1"],
        "ids": ["doc1/0/pl", "doc1/1/pl", "doc2/0/pl"],
    }

    mock_splitter_instance.split_text.assert_any_call("Content 1")
    mock_splitter_instance.split_text.assert_any_call("Content 2")
    assert result == expected


@patch("template_pipelines.steps.doc_vectorization.ingestion.json")
@patch("template_pipelines.steps.doc_vectorization.ingestion.NamedTemporaryFile")
@patch.object(Ingestion, "load", new_callable=AsyncMock)
@patch.object(Ingestion, "load_document_async", new_callable=AsyncMock)
def test_ingestion_run(
    mock_load_document_async, mock_load, mock_named_tempfile, mock_json, fixture_ingestion
):
    """Test the Ingestion run method."""
    mock_input_dir = MagicMock()
    mock_input_dir.glob.return_value = ["file1.md", "file2.md"]
    fixture_ingestion.inputs = {"doc_input_dir": mock_input_dir}
    mock_load_document_async.side_effect = ["document 1", "document 2"]
    mock_load.return_value = {"metadatas": [], "texts": [], "ids": []}
    fixture_ingestion.outputs = {}

    fixture_ingestion.run()

    mock_input_dir.glob.assert_called_once_with("**/*.md")
    mock_load_document_async.assert_any_call("file1.md")
    mock_load_document_async.assert_any_call("file2.md")
    mock_load.assert_called_once()
    mock_named_tempfile.assert_called_once()
    mock_json.dump.assert_called_once_with(
        mock_load.return_value, mock_named_tempfile.return_value.__enter__()
    )
    assert "aggs.json" in fixture_ingestion.outputs
