"""Base step for using OpenAI and Pyrogai functionalities."""

import os

from aif.pyrogai.steps.step import Step
from template_pipelines.utils.hello_gen_ai.azure_openai_client import AzureOpenAIClient


class BaseGenAIStep(Step):
    """Define a customized base step for using OpenAI."""

    def __init__(self):
        """Instatiate genai_client with required authentication."""
        super().__init__()
        headers = self.config["gen_ai"]["headers"]
        openai_settings = {
            "genai_proxy": self.config["gen_ai"]["genai_proxy"],
            "cognitive_services": self.config["gen_ai"]["cognitive_services"],
            "open_api_version": self.config["gen_ai"]["open_api_version"],
            "headers": headers,
        }
        self.chat_engine = self.config["gen_ai"]["chat_engine"]
        self.embedding_engine = self.config["gen_ai"]["embedding_engine"]

        if self.secrets.get("AML-APP-SP-SECRET"):
            os.environ["AZURE_TENANT_ID"] = self.secrets["tenant-id"]
            os.environ["AZURE_CLIENT_ID"] = self.secrets["AML-APP-SP-ID"]
            os.environ["AZURE_CLIENT_SECRET"] = self.secrets["AML-APP-SP-SECRET"]

        self.genai_client = AzureOpenAIClient(
            openai_settings, self.chat_engine, self.embedding_engine
        )
