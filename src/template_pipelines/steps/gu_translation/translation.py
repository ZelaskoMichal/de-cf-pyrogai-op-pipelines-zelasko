"""GU Translation step class."""

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from template_pipelines.utils.gu_translation.base_genai_step import BaseGenAIStep


# Define Translation class and inherit properties from a customized BaseGenAIStep class
# This enables both Pyrogai and GenAI functionalities to be utilized simultaneously
class Translation(BaseGenAIStep):
    """Translation step."""

    def run_multithreaded(self, func, func_inputs, max_workers):
        """Execute a function concurrently with multiple threads and write results to a dataframe."""
        text, original, targets = func_inputs
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(func, text, original, target): target for target in targets}

        df = pd.DataFrame({"language": [original], "translation": [text]})
        index = 1
        for future in as_completed(futures):
            target = futures[future]
            translated_text = future.result()
            df.loc[index] = [target, translated_text]
            index += 1

        return df

    # Pyrogai executes code defined under run method
    def run(self):
        """Run GU Translation."""
        # Log information using self.logger.info from Pyrogai Step class
        self.logger.info("Running GU translation...")

        # Set translation inputs and send threaded requests
        # The run_multithreaded function stores the response results in a dataframe
        text = self.runtime_parameters["text"]
        original = self.runtime_parameters["original_language"]
        targets = self.runtime_parameters["target_languages"].split(",")
        thread_num = self.config["gu_translation"]["thread_num"]
        df = self.run_multithreaded(self.translate, (text, original, targets), thread_num)

        self.logger.info(f"The text: '{text}' has been translated into following languages:\n{df}")
        self.logger.info("Translation is done.")
