"""Tests for toolkit."""

import unittest
from unittest.mock import AsyncMock, MagicMock

import pytest

from template_pipelines.utils.doc_vectorization.toolkit import QuestionGenerationEmbedding


class TestQuestionGenerationEmbedding(unittest.TestCase):
    """Test cases for the QuestionGenerationEmbedding class."""

    def setUp(self):
        """Set up a test environment before each test method is executed."""
        self.mock_chain = MagicMock()
        self.mock_embedding = MagicMock()
        self.qge = QuestionGenerationEmbedding(self.mock_chain, self.mock_embedding)

    def tearDown(self):
        """Remove the test environment after each test method is executed."""

    def test__init__(self):
        """Test the initialization of the QuestionGenerationEmbedding class."""
        self.assertEqual(self.qge.question_generation_chain, self.mock_chain)
        self.assertEqual(self.qge.embedding_model, self.mock_embedding)

    def test_dict_to_test(self):
        """Test _dict_to_text."""
        qa_pairs = [
            {"question": "What is the capital of Vietnam?", "answer": "Ha Noi."},
            {"question": "Can you name one traditional Vietnamese dish?", "answer": "Pho."},
        ]
        expected = "What is the capital of Vietnam? Ha Noi. Can you name one traditional Vietnamese dish? Pho."
        result = self.qge._dict_to_text(qa_pairs)
        self.assertEqual(result, expected)

    def test_embed_documents(self):
        """Test embed_documents."""
        self.mock_chain.run.side_effect = [
            [{"question": "Q1", "answer": "A1"}],
            [{"question": "Q2", "answer": "A2"}],
        ]
        self.mock_embedding.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]

        texts = ["Text1", "Text2"]
        result = self.qge.embed_documents(texts)

        self.mock_chain.run.assert_any_call("Text1")
        self.mock_chain.run.assert_any_call("Text2")
        self.mock_embedding.embed_documents.assert_called_once_with(["Q1 A1", "Q2 A2"])
        self.assertEqual(result, [[0.1, 0.2], [0.3, 0.4]])

    def test_embed_query(self):
        """Test embed_query."""
        text = "I love Pho!"
        self.mock_embedding.embed_query.return_value = [[0.1, 0.2]]
        result = self.qge.embed_query(text)

        self.mock_embedding.embed_query.assert_called_once_with(text)
        self.assertEqual(result, [[0.1, 0.2]])

    @pytest.mark.asyncio
    async def test_aembed_documents(self):
        """Test aembed_documents."""
        self.mock_chain.arun = AsyncMock()
        self.mock_chain.arun.side_effect = [
            [{"question": "Q1", "answer": "A1"}],
            [{"question": "Q2", "answer": "A2"}],
        ]
        self.mock_embedding.aembed_documents = AsyncMock()
        self.mock_embedding.aembed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]

        texts = ["Text1", "Text2"]
        result = await self.qge.aembed_documents(texts)

        self.mock_chain.arun.assert_any_call("Text1")
        self.mock_chain.arun.assert_any_call("Text2")
        self.mock_embedding.aembed_documents.assert_called_once_with(["Q1 A1", "Q2 A2"])
        self.assertEqual(result, [[0.1, 0.2], [0.3, 0.4]])

    @pytest.mark.asyncio
    async def test_aembed_query(self):
        """Test aembed_query."""
        text = "I love Pho!"
        self.mock_embedding.aembed_query = AsyncMock()
        self.mock_embedding.aembed_query.return_value = [[0.1, 0.2]]
        result = await self.qge.aembed_query(text)

        self.mock_embedding.aembed_query.assert_called_once_with(text)
        self.assertEqual(result, [[0.1, 0.2]])
