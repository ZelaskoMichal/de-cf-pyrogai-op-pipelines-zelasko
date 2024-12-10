"""Resuable methods and classes."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)


def run_multithreaded(func, func_inputs, max_workers, output_path):
    """Execute a function concurrently with multiple threads and writes results to a file."""
    query, df = func_inputs
    with ThreadPoolExecutor(max_workers=max_workers) as executor, open(output_path, "w") as f:
        futures = {executor.submit(func, query, row): row for _, row in df.iterrows()}

        for future in as_completed(futures):
            result = future.result()
            f.write(f"{result[0]}; {result[1]}\n")


@dataclass
class VideoSummaryPrompt:
    """A dataclass representing a prompt for analyzing and summarizing product influencer videos."""

    prompt: str = """
        You are an expert marketer reviewing product influencer videos.
        You want to understand how effective this video is likely to be, and \
        if the influencer is using my products appropriately.

        You will perform the following tasks:
        1. Identify the specific name of the main product.
        2. Describe how the main product is being used or demonstrated or talked about.
        3. Identify the benefits of how the main product was presented.
        4. Identify the drawbackas of the main product was presented.
        5. Recommend some improvement on how the main product can be presented.

        **Response Format**:
        - Provide all 5 answers in **one single line**, separated by **exactly 4 semicolons (;)**.
        - Keep each answert short and concise.
        - The response must contain **exactly 5 parts**, no more, no less.
        - Ensure the total response does **not exceed 200 characters**.
        - Do not use newlines or any additional semicolons.

        **Example**:
        Shampoo X; applied during hair wash; easy application; unclear hair type benefit; show more diverse hair types.
        """
