"""Base step for using GenAI and Pyrogai functionalities."""

import os

from langchain.chains import QAGenerationChain
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from aif.genai_utils import GenAIToken
from aif.pyrogai.steps.step import Step


class BaseGenAIStep(Step):
    """Define a customized base step for using GenAI."""

    def __init__(self):
        """Instatiate GenAI utilities with required authentication."""
        super().__init__()

        if self.secrets.get("AML-APP-SP-SECRET"):
            os.environ["AZURE_TENANT_ID"] = self.secrets["tenant-id"]
            os.environ["AZURE_CLIENT_ID"] = self.secrets["AML-APP-SP-ID"]
            os.environ["AZURE_CLIENT_SECRET"] = self.secrets["AML-APP-SP-SECRET"]

        self.genai_token = GenAIToken()

        config = self.config["doc_vectorization"]
        self.chat_model = AzureChatOpenAI(
            azure_endpoint=config["genai_proxy"],
            azure_ad_token_provider=self.genai_token.token,
            api_version=config["open_api_version"],
            default_headers=config["headers"],
            **config["llm_params"],
        )
        self.embedding_model = AzureOpenAIEmbeddings(
            azure_endpoint=config["genai_proxy"],
            azure_ad_token_provider=self.genai_token.token,
            api_version=config["open_api_version"],
            default_headers=config["headers"],
            **config["embedding_params"],
        )
        self.qa_ge_chain = QAGenerationChain.from_llm(self.chat_model)

        self.logger.info("BaseGenAIStep is in use.")
