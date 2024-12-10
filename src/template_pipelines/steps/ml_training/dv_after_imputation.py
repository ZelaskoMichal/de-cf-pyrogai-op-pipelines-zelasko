"""Customized data validation class.

This validates transformed data from imputation based on expectations
defined in the config file and then visualizes validation results.
In future pyrogai releases, the visualization of validation results will be included in the
standard DqStep by default that can be used instead as a standalone step (zero code solution).
"""

from aif.pyrogai.const import Platform
from aif.pyrogai.steps import DqStep


# Define DataValidationAfterImputation class
# and inherit properties from pyrogai DqStep class
class DataValidationAfterImputationDqStep(DqStep, suffix="DqStep"):  # type: ignore
    """Data Validation After Imputation step."""

    # Pyrogai executes code defined under run method
    def run(self):
        """Run Data Validation After Imputation step."""
        # A temporary workaround to visualize great expectation results
        # Some default GE visualization feature might be in next Pyrogai releases
        self.run_checks(step_name=self.step_name)
        self.generate_data_docs()
        self.raise_for_status()

        # Find a GE warning.html and set it to ioslot to visualize results
        ge_doc_path = next(self.ioctx.get_fns(f"ge_dq_datadocs/{self.step_name}/**/warning.html"))
        if self.platform in (Platform.AML, Platform.DBR):
            self.outputs["mlflow_ge_doc"] = ge_doc_path
        elif self.platform == Platform.VERTEX:
            with open(ge_doc_path, "r") as infile:
                html = infile.read()
                self.outputs["kfp_ge_doc"] = html
