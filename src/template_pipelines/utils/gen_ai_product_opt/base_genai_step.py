"""Base step for using OpenAI and Pyrogai functionalities."""

import os

from aif.pyrogai.steps.step import Step
from template_pipelines.utils.gen_ai_product_opt.toolkit import AzureOpenAIClient


class BaseGenAIStep(Step):
    """Define a customized base step for using OpenAI."""

    def __init__(self):
        """Instatiate genai_client with required authentication."""
        super().__init__()
        config = self.config["gen_ai_product_opt"]
        openai_settings = {
            "genai_proxy": config["genai_proxy"],
            "cognitive_services": config["cognitive_services"],
            "open_api_version": config["open_api_version"],
            "headers": config["headers"],
            "temperature": config["temperature"],
            "chat_engine": config["chat_engine"],
            "embedding_engine": config["embedding_engine"],
            "token_refresh_thre": config["token_refresh_thre"],
        }

        if self.secrets.get("AML-APP-SP-SECRET"):
            os.environ["AZURE_TENANT_ID"] = self.secrets["tenant-id"]
            os.environ["AZURE_CLIENT_ID"] = self.secrets["AML-APP-SP-ID"]
            os.environ["AZURE_CLIENT_SECRET"] = self.secrets["AML-APP-SP-SECRET"]

        self.genai_client = AzureOpenAIClient(openai_settings)
        self.logger.info("BaseGenAIStep is in use.")
