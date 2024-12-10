"""Cosine Similarity step class."""

from operator import itemgetter

import numpy as np
import pandas as pd
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

from template_pipelines.utils.gen_ai_product_opt.base_genai_step import BaseGenAIStep
from template_pipelines.utils.gen_ai_product_opt.prompts import ProductSummaryPrompt
from template_pipelines.utils.gen_ai_product_opt.toolkit import (
    VectoreStoreRetrieverFilter,
    create_chat_prompt,
)


# Define CosineSimilarity class and inherit properties a from customized BaseGenAIStep class
# This enables both Pyrogai and GenAI/OpenAI functionalities to be utilized simultaneously
# e.g. BaseGenAIStep instantiates self.genai_client which uses self.config and self.secrets for authentication
class CosineSimilarity(BaseGenAIStep):
    """Cosine Similarity step."""

    def format_docs(self, docs):
        """Create a string by joining a list of docs."""
        return ",".join([d.page_content for d in docs])

    def create_documents(self, text, page):
        """Create a list of documents from text."""
        docs = text.split(",")
        return [Document(page_content=doc, metadata=dict(page=page)) for doc in docs]

    def create_doc_array(self, df, column):
        """Create a doc array by concatenating multiple docs."""
        return np.concatenate(
            df.apply(lambda row: self.create_documents(row[column], row.name), axis=1)
        )

    # Pyrogai executes code defined under run method
    def run(self):
        """Run Cosine Similarity step."""
        # Log information using self.logger.info from Pyrogai Step class
        self.logger.info("Running cosine similarity calculation...")

        # Get a full filepath to read a file created in the Preprocessing step
        # and stored in a shared location using Pyrogai Step self.ioctx.get_fn
        input_path = self.ioctx.get_fn("data_preprocessing/preprocessed.parquet")
        df = pd.read_parquet(input_path)

        # Create a product summary prompt for chat_model
        chat_prompt = create_chat_prompt(
            ProductSummaryPrompt.prompt, ProductSummaryPrompt.user_prompt
        )

        # Use LCEL to build a chat_chain of LangChain components
        # chat_chain in parallel processes inputs and return two outputs
        chat_chain = chat_prompt | self.genai_client.chat_model | StrOutputParser()
        chat_chain = RunnableParallel(product_summary=chat_chain, page=itemgetter("page"))

        # Create a vectorstore for product keyword embeddings
        self.logger.info("Adding product keywords to vectorstore...")
        doc_array = self.create_doc_array(df, "keywords")
        config = self.config["gen_ai_product_opt"]
        retriever = VectoreStoreRetrieverFilter(
            vectorstore=FAISS.from_documents(
                doc_array,
                self.genai_client.embedding_model,
                distance_strategy=config["distance_strategy"],
            ),
            search_kwargs={
                "score_threshold": config["similarity_threshold"],
                "k": config["num_results"],
            },
        )

        # Build final_chain which generates a product summary and then finds product keywords relevant to the summary
        final_chain = (
            chat_chain
            | {"query": itemgetter("product_summary"), "_filter": {"page": itemgetter("page")}}
            | retriever
            | self.format_docs
        )

        # Apply final_chain row wise to df
        self.logger.info("Find relevant product keywords...")
        df["updated_keywords"] = df.apply(
            lambda row: final_chain.invoke(
                {
                    "current_title": row["title"],
                    "list_features": row["product_description"],
                    "page": row.name,
                }
            ),
            axis=1,
        )

        # Save updated product keywords to a shared work directory using ioctx
        output_path = self.ioctx.get_output_fn("cosine_similarity/updated_product_keywords.parquet")
        df.to_parquet(output_path)

        self.logger.info("Cosine similarity calculation is done.")
