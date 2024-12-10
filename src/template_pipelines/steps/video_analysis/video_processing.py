"""Video Processing step class."""

import time

import pandas as pd

from template_pipelines.utils.video_analysis.base_genai_step import BaseGenAIStep
from template_pipelines.utils.video_analysis.toolkit import VideoSummaryPrompt, run_multithreaded


# Define VideoProcessing class and inherit properties a from customized BaseGenAIStep class
# This enables both Pyrogai and GenAI functionalities to be utilized simultaneously
class VideoProcessing(BaseGenAIStep):
    """Video Processing step."""

    # Pyrogai executes code defined under run method
    def run(self):
        """Run Video Processing."""
        # Log information using self.logger.info from Pyrogai Step class
        self.logger.info("Running video processing...")

        # Read in ioslot-defined data
        df = pd.read_csv(self.inputs["video_data.csv"], sep=";")

        # Set number of threads used to send requests to the Gemini model
        thread_num = self.config["video_analysis"]["thread_num"]

        # Send threaded requests and measure the processing time
        # The run_multithreaded function stores the response results in responses.txt
        # Which is later used in the result_aggregation step
        start = time.time()
        output_path = self.ioctx.get_output_fn("video_processing/responses.txt")
        run_multithreaded(self.ask_gemini, (VideoSummaryPrompt.prompt, df), thread_num, output_path)

        self.logger.info(f"Video processing time: {time.time()-start}")
        self.logger.info("Video processing is done.")
