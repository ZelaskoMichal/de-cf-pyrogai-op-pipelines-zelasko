"""ValidateCrunchOutputs step class."""
from aif.pyrogai.steps.step import Step  # noqa: E402
from template_pipelines.utils.crunch_tutorial.utils import pretty_list_files


class ValidateCrunchOutputs(Step):
    """ValidateCrunchOutputs step."""

    def run(self):
        """Run preprocessing step."""
        self.logger.info(f"Running {self.step_name} step..")

        local_tmp = self.inputs["sales_data_output"]

        pretty_list_files(local_tmp)

        self.logger.info("CRUNCH outputs have been validated")
