"""Step that allows DQStep to save files to ioslots."""

from tempfile import NamedTemporaryFile

import pandas as pd

from aif.pyrogai.steps import DqStep
from template_pipelines.utils.optimization.io_utils import load_tables


class OutputWithSaveToIoslotsDqStep(DqStep, suffix="DqStep"):  # type: ignore
    """This step copies data to ioslots after running Great Expectations."""

    def run(self) -> None:
        """Copy outputs from ioctx dir to ioslot dir and copy output.parquet to output.csv also.

        The directory copy outputs in the same format saved to ioctx (typically parquet).
        To demonstrate, we also save as csv.
        """
        # standard run
        self.run_checks()
        self.generate_data_docs()
        self.raise_for_status()

        # add the output.warning html report to output_ge_warning.html
        self.copy_ge_report_to_custom_dir("output.warning", "output_ge_warning.html")

        # custom output
        # in run rather than post_run because we only want to do this if no exception occurs
        # save dir of parquet files
        self.outputs["output_dir"] = self.ioctx.get_output_fn(self.config["output_tmp_dir"])
        self.logger.debug("Saved ioslot output_dir")
        # save single table in csv format
        df = load_tables(self.ioctx.get_fns(f"{self.config['output_tmp_dir']}/*.parquet"))[0][
            "output"
        ]
        self.save_output(df)

    def save_output(self, output_df: pd.DataFrame) -> None:
        """Save output table to ioslot."""
        with NamedTemporaryFile() as f:
            output_df.to_csv(f.name)
            self.outputs["output.csv"] = f.name
            self.logger.debug("Saved ioslot output.csv")

    def copy_ge_report_to_custom_dir(self, expectation_name: str, ioslot_name: str) -> None:
        """Add the GE html report to a custom directory in IOSlots.

        This needs to run after the generate_data_docs() method.

        Args:
            expectation_name (str): expectation name as in the config.json file, e.g. "output.warning"
            ioslot_name (str): name of the IOSlot defined in the pipeline.yml file
        """
        name, level = expectation_name.split(".")
        # gets the path where the expectation report should have been created
        ge_doc_path = next(
            self.ioctx.get_fns(f"ge_dq_datadocs/{self.step_name}/{name}/**/{level}.html")
        )
        self.outputs[ioslot_name] = ge_doc_path
