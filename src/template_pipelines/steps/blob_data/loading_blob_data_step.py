"""Loading data step for Blob data."""

import os

from aif.pyrogai.steps.step import Step  # noqa: E402


class LoadingBlobDataStep(Step):
    """Class for getting Blob data from AML and DBR."""

    def load_from_dbr(self):
        """Load data from DBR."""
        from pyspark.sql import SparkSession

        if not self.runtime_parameters.get("dbr_table"):
            raise RuntimeError("Lack of 'dbr_table' parameter in runtime parameters.")

        spark = SparkSession.builder.getOrCreate()
        df = spark.table(self.runtime_parameters["dbr_table"])
        self.logger.info(f"DataFrame loaded. Schema: {df.schema.simpleString()}")

        return df

    def load_from_aml(self):
        """Load data from AML."""
        folder_path = self.runtime_parameters["folder_path"]
        file_path = self.runtime_parameters["file_path"]

        if (folder_path != "" and file_path != "") or (folder_path == "" and file_path == ""):
            raise RuntimeError(
                "Either parameter 'folder_path' or 'file_path' must be provided, but not both or neither."
            )

        if file_path != "":
            blob_file = self.inputs["blob_file"]
            return self.load_parquet_file(blob_file)
        elif folder_path != "":
            blob_folder = self.inputs["blob_folder"]
            return self.combine_parquet_files(blob_folder)

    def load_from_local(self):
        """Load data from Local."""
        # If you want to run the pipeline locally but fetch data directly from DBR,
        # you can change the data loading method:
        # Use self.load_from_dbr(). Remember to set up the `dbr_table` parameter:
        # params:
        #   dbr_table: "default.test_table" # noqa E800
        #
        # Additionally, ensure your 'databricks-connect' configuration
        # is correctly set up to enable this functionality.

        return self.load_from_aml()

    def load_parquet_file(self, file_path):
        """Load a single Parquet file."""
        import pyarrow.parquet as pq

        self.logger.info(f"Loading Parquet file from: {file_path}")
        table = pq.read_table(file_path)
        self.logger.info(f"Loaded Parquet file. Columns: {table.column_names}")

        return table

    def combine_parquet_files(self, folder_path):
        """Combine all Parquet files in a folder and its subfolders into a single DataFrame."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.logger.info(f"Combining Parquet files from folder and subfolders: {folder_path}")
        files = []
        for root, _, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.endswith(".snappy.parquet"):
                    files.append(os.path.join(root, filename))

        self.logger.info(f"Found {len(files)} Parquet files")
        tables = [pq.read_table(file) for file in files]
        combined_table = pa.concat_tables(tables)
        self.logger.info(f"Combined Parquet DataFrame. Columns: {combined_table.column_names}")

        return combined_table

    def run(self):
        """Run method."""
        self.logger.info(f"Getting data from Blob on {self.platform}")
        self.logger.warning(self.runtime_parameters)

        if self.platform == "DBR":
            df = self.load_from_dbr()
            self.logger.info(df.columns)
        elif self.platform == "AML":
            df = self.load_from_aml()
            self.logger.info(df.column_names)
        elif self.platform == "Local":
            df = self.load_from_local()
            self.logger.info(df.column_names)
        else:
            raise RuntimeError(
                f"Wrong platform selected: {self.platform}. Valid platforms are 'DBR', 'Local', or 'AML'."
            )

        self.logger.info(f"Finish loading data from Blob.")
