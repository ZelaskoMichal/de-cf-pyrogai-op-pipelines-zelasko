"""Hello world for gen AI template."""

from template_pipelines.utils.hello_gen_ai.base_genai_step import BaseGenAIStep


class HelloGenAI(BaseGenAIStep):
    """HelloGenAI step."""

    def run(self):
        """Run HelloGenAI step."""
        self.logger.info("Running HelloGenAI step...")

        message = {
            "system": self.runtime_parameters.get("system_message"),
            "human": self.runtime_parameters.get("user_message"),
        }

        response = self.genai_client.get_chat_response(
            message, temperature=self.config["gen_ai"]["temperature"]
        )
        self.logger.info(response)

        response_embedded = self.genai_client.get_embedding(response.split(" "))
        self.logger.info(len(response_embedded))

        self.logger.info("Finish HelloGenAI step...")
