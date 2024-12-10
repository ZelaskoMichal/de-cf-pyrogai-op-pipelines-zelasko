"""Base step for using GenAI and Pyrogai functionalities."""

import os

import httpx

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
        self.logger.info("BaseGenAIStep is in use.")

    def translate(self, text, original, target):
        """Send a query for translation to the GenAI platform."""
        url = self.config["gu_translation"]["genai_url"]
        service_endpoint = self.config["gu_translation"]["service_endpoint"]
        final_url = f"{url}{service_endpoint}"
        headers = self.config["gu_translation"]["headers"]
        headers["Authorization"] = f"Bearer {self.genai_token.token()}"

        body = {
            "text_to_translate": text,
            "source_language": original,
            "target_language": target,
            "expected_tone": "official",
            "list_of_terms": {"GU": "Generalized Utility"},
        }

        try:
            with httpx.Client() as client:
                response = client.post(final_url, headers=headers, json=body, timeout=60)
                response.raise_for_status()
                return response.json()["translation"]
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error for the translation from {original} to {target}: {e}")
        except Exception as e:
            self.logger.error(f"Error for the translation from {original} to {target}: {e}")
        return "Error during translation"
