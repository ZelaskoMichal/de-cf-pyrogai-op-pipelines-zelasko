"""Copy input data from ioslots to ioctx."""

from pathlib import Path

from aif.pyrogai.steps.step import Step
from template_pipelines.utils.optimization.io_utils import copy_data_to_parquet


class CopyInputToIoctx(Step):
    """This step copies data from an ioslot to ioctx.

    It can copy 3 types of files: parquet, xls or xlsx, csv and then paste them as parquet files to
    ioctx. It also uses read function for files with "NA" in them (in case of csv or Excel files).
    This step was created because Great Expectations data quality step can't handle CSV files with
    "NA" strings (treats them like nulls) or multi-tab Excel files.
    """

    def run(self):
        """Copy inputs to ioctx dir."""
        # needed due to self.inputs.values not enabled
        ioslots_file_paths = [self.inputs[key] for key in self.inputs.keys()]
        dest_ioctx_path = Path(self.ioctx.get_output_fn(self.config["input_tmp_dir"]))
        copy_data_to_parquet(
            file_paths=ioslots_file_paths,
            dest_path=dest_ioctx_path,
        )
