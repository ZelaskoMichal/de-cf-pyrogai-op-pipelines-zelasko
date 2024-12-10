"""Vectorization step class."""

import asyncio
import json
from tempfile import TemporaryDirectory

import faiss
from langchain.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from template_pipelines.utils.doc_vectorization.base_genai_step import BaseGenAIStep
from template_pipelines.utils.doc_vectorization.toolkit import QuestionGenerationEmbedding


class Vectorization(BaseGenAIStep):
    """Vectorization step."""

    def __init__(self):
        """Set up vector store for the class using a question-answer generation chain and an embedding model."""
        super().__init__()
        qa_embedding = QuestionGenerationEmbedding(
            question_generation_chain=self.qa_ge_chain, embedding_model=self.embedding_model
        )
        self.vectorstore = FAISS(
            embedding_function=qa_embedding,
            index=faiss.IndexFlatL2(1536),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    async def add_batch_async(self, batch):
        """Asynchronously add a batch of texts, metadata, and IDs to the FAISS vector store."""
        await self.vectorstore.aadd_texts(**batch)

    async def process_texts_in_batches_async(self, text_details, batch_size=10):
        """Asynchronously process a collection of text documents in batches and add them to the vector store."""
        texts = text_details["texts"]
        metadatas = text_details["metadatas"]
        ids = text_details["ids"]

        batches = [
            {
                "texts": texts[i : i + batch_size],  # noqa: E203
                "metadatas": metadatas[i : i + batch_size],  # noqa: E203
                "ids": ids[i : i + batch_size],  # noqa: E203
            }
            for i in range(0, len(metadatas), batch_size)
        ]

        await asyncio.gather(*[self.add_batch_async(batch) for batch in batches])

    def run(self):
        """Run Vectorization."""
        self.logger.info("Running vectorization...")

        with open(self.inputs["aggs.json"], "rb") as f:
            aggs = json.load(f)

        for key in aggs:
            self.logger.info(f"{key}: {len(aggs[key])}")

        asyncio.run(self.process_texts_in_batches_async(aggs, batch_size=10))

        with TemporaryDirectory() as tmpd:
            self.vectorstore.save_local(tmpd)
            self.outputs["faiss_vector_db"] = tmpd

        self.logger.info("Vectorization is done.")
