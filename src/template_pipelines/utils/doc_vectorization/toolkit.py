"""Resuable methods and classes."""

from typing import List

from langchain.chains.base import Chain
from langchain.embeddings.base import Embeddings


class QuestionGenerationEmbedding(Embeddings):
    """Class to generate question and embed the question instead of the orignal source."""

    question_generation_chain: Chain
    embedding_model: Embeddings

    def __init__(self, question_generation_chain, embedding_model):
        """Initialize the QuestionGenerationEmbedding class with a question generation chain and an embedding model."""
        self.question_generation_chain = question_generation_chain
        self.embedding_model = embedding_model

    def _dict_to_text(self, qa_pairs=List):
        """Converts a dictionary of question-answer pairs to a string."""
        text = ""
        for qa_pair in qa_pairs:
            if text != "":
                text += " "
            text += qa_pair["question"] + " " + qa_pair["answer"]
        return text

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        transformed_texts = [
            self._dict_to_text(self.question_generation_chain.run(text)) for text in texts
        ]
        return self.embedding_model.embed_documents(transformed_texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embedding_model.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        transformed_texts = [
            self._dict_to_text(await self.question_generation_chain.arun(text)) for text in texts
        ]
        return await self.embedding_model.aembed_documents(transformed_texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return await self.embedding_model.aembed_query(text)
