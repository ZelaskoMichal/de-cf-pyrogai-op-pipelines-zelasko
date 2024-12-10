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

    def ask_gemini(self, query, input_dict):
        """Send a query and a video file URI to the GenAI platform."""
        self.logger.info(f"{input_dict['id']}. {input_dict['genai_filepath']}")

        url = self.config["video_analysis"]["genai_url"]
        model_endpoint = self.config["video_analysis"]["model_endpoint"]
        final_url = f"{url}{model_endpoint}"
        headers = self.config["video_analysis"]["headers"]
        headers["Authorization"] = f"Bearer {self.genai_token.token()}"

        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": query},
                        {
                            "fileData": {
                                "mimeType": "video/mp4",
                                "fileUri": input_dict["genai_filepath"],
                            }
                        },
                    ],
                }
            ]
        }

        try:
            # Run the request
            with httpx.Client() as client:
                response = client.post(final_url, headers=headers, json=body, timeout=60)
                response.raise_for_status()
                output = response.json()["candidates"][0]["content"]["parts"][0]["text"].replace(
                    "\n", ""
                )
                return (input_dict["id"], output)
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error for {input_dict['genai_filepath']}: {e}")
        except Exception as e:
            self.logger.error(f"Error for {input_dict['genai_filepath']}: {e}")
        return (input_dict["id"], None)
