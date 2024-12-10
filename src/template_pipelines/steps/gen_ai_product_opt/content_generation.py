"""Content Generation class."""

from tempfile import NamedTemporaryFile

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

from template_pipelines.utils.gen_ai_product_opt.base_genai_step import BaseGenAIStep
from template_pipelines.utils.gen_ai_product_opt.prompts import (
    BulletPointPrompt,
    ProductTitlePrompt,
)
from template_pipelines.utils.gen_ai_product_opt.toolkit import create_chat_prompt


# Define ContentGeneration class and inherit properties a from customized BaseGenAIStep class
# This enables both Pyrogai and GenAI/OpenAI functionalities to be utilized simultaneously
# e.g. BaseGenAIStep instantiates self.genai_client which uses self.config and self.secrets for authentication
class ContentGeneration(BaseGenAIStep):
    """Content Generation step."""

    # Pyrogai executes code defined under run method
    def run(self):
        """Run content generation step."""
        # Log information using self.logger.info from Pyrogai Step class
        self.logger.info("Running content generation...")

        # Get a full filepath to read a file created in the Cosine Similarity step
        # and stored in a shared location using Pyrogai Step self.ioctx.get_fn
        input_path = self.ioctx.get_fn("cosine_similarity/updated_product_keywords.parquet")
        data = pd.read_parquet(input_path)

        # Create product title and description prompts for chat_model
        chat_product_title_prompt = create_chat_prompt(
            ProductTitlePrompt.prompt, ProductTitlePrompt.user_prompt
        )
        chat_product_desc_prompt = create_chat_prompt(
            BulletPointPrompt.prompt, BulletPointPrompt.user_prompt
        )

        # Use LCEL to build product title and description chains of LangChain components
        product_title_chain = (
            chat_product_title_prompt | self.genai_client.chat_model | StrOutputParser()
        )
        product_desc_chain = (
            chat_product_desc_prompt | self.genai_client.chat_model | StrOutputParser()
        )

        # Build parallel_optimized_chain to perform two chain operations in parallel
        # This chain generates optimized product title and description strings
        parallel_optimized_chain = RunnableParallel(
            optimized_title=product_title_chain, optimized_description=product_desc_chain
        )

        # Apply parallel_optimized_chain row wise to data
        self.logger.info("Generating optimized product titles and descriptions...")
        optimized_content = data.apply(
            lambda row: parallel_optimized_chain.invoke(
                {
                    "product_title": row["title"],
                    "list_keywords": row["updated_keywords"],
                    "current_bullet_points": row["product_description"],
                }
            ),
            axis=1,
            result_type="expand",
        )
        for column in optimized_content.columns:
            data[column] = optimized_content[column]

        # Write data into ioslot-defined output
        with NamedTemporaryFile() as f:
            data.to_parquet(f.name)
            self.outputs["optimized_product_description.parquet"] = f.name
            self.logger.info("Optimized titles and descriptions have been stored.")

        self.logger.info("Content generation is done.")
