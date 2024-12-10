"""Example tutorial CRUNCH step class."""
from pathlib import Path

import pandas as pd

from aif.pyrogai.steps.crunch.crunch_dummy_config import crunch_step_config, logger
from aif.pyrogai.steps.crunch.crunch_step import CrunchStep
from template_pipelines.utils.crunch_tutorial.example_logic import process_data


class TutorialCrunchStep(CrunchStep):
    """Example tutorial CRUNCH step class."""

    @staticmethod
    def crunch_run():
        """Run CRUNCH job."""
        input_path: Path = crunch_step_config.input
        output_path: Path = crunch_step_config.output

        logger.info(
            "Hello from CRUNCH job! I am running in a pod in CRUNCH AKS cluster. "
            "There are many jobs like me, as many as there are subdirectories in your input slot."
        )

        runtime_param: dict = crunch_step_config.runtime_parameters["my_runtime_param"]
        config_param: dict = crunch_step_config.config["my_config_param"]

        # The log messages are available via CRUNCH web UI, you will see a link printed
        # to console when you run your pipeline.
        logger.info(f"Local input path is: {input_path}")
        logger.info(f"Local output path is: {output_path}")

        logger.info(f"My runtime parameter is: {runtime_param}")
        logger.info(f"My config parameter is: {config_param}")

        # Secrets configured in config_crunch-tutorial.yaml are available
        logger.info(f"My secret parameters are: {list(crunch_step_config.secrets.keys())}")

        logger.debug(
            f"This debug message is visible in CRUNCH job logs if you run your pipeline with --debug flag"
        )

        pd.DataFrame()  # pandas is available, requirements were installed,  see config_crunch-tutorial.yaml

        process_data(input_path, output_path)  # example logic, replace with yours

    def do_not_add_methods_here(self):
        """This method is not available in CRUNCH job. Write your code in crunch_run method or import it from src/."""


def do_not_add_methods_here_either():
    """This method is not available in CRUNCH job. Write your code in crunch_run method or import it from src/."""
