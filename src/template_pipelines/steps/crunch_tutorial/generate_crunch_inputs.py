"""GenerateCrunchInputs step class."""
import os
from tempfile import TemporaryDirectory

from aif.pyrogai.steps.step import Step  # noqa: E402
from template_pipelines.utils.crunch_tutorial.generate_data import generate_dataset


class GenerateCrunchInputs(Step):
    """GenerateCrunchInputs step class."""

    def run(self):
        """Run GenerateCrunchInputs step."""
        self.logger.info(f"Running {self.step_name} step..")
        self.logger.info("Generating CRUNCH input data..")

        with TemporaryDirectory() as tmpd:
            generate_dataset(tmpd)
            self.logger.info([x for x in os.walk(tmpd)])
            self.outputs["sales_data"] = tmpd

        self.logger.info("CRUNCH input data have been generated and uploaded")
