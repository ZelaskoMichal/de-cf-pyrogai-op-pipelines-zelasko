"""Preprocessing step class."""

from pathlib import Path

import pandas as pd

from aif.pyrogai.steps.step import Step  # noqa: E402


# Define Preprocessing class and inherit properties from pyrogai Step class
class Preprocessing(Step):
    """Preprocessing step."""

    def read_data_into_dict(self):
        """Read ioslot-defined data into a dictionary for easier pre-processing."""
        tables = {}
        for name, _ in self.inputs.items():
            self.logger.info(f"Reading in {name} ...")
            # Get basename without extension
            basename = Path(name).stem
            tables[basename] = pd.read_parquet(self.inputs[name])
        return tables

    def set_datatype(self, tables, datatype):
        """Set one datatype for all tables."""
        for table_name in tables:
            tables[table_name] = tables[table_name].astype(datatype)
        return tables

    def combine_text_columns(self, table, regex, delimeter, new_column):
        """Combine multiple text columns using regex for column selection."""
        table[new_column] = table.filter(regex=regex).apply(lambda x: delimeter.join(x), axis=1)
        return table

    def transform_text(self, table, column):
        """Remove a newline character and convert to lowercase letters."""
        table[column] = table[column].apply(lambda x: x.replace("\n", "").lower())
        return table

    def left_join_all(self, tables, key):
        """Left join all tables."""
        table_list = list(tables.values())
        main_table = table_list[0]
        for table in table_list[1:]:
            main_table = main_table.merge(table, on=key, how="left")
        return main_table

    # Pyrogai executes code defined under run method
    def run(self):
        """Run preprocessing step."""
        # Log information using self.logger.info from Pyrogai Step class
        self.logger.info("Running preprocessing...")

        # Read in ioslot-defined data
        tables = self.read_data_into_dict()
        self.logger.info("Data has been read sucessfully.")

        # Set appropriate data types
        tables = self.set_datatype(tables, str)

        # Comnbine multiple product description texts into one
        tables["advertised_products"] = self.combine_text_columns(
            tables["advertised_products"], "desc", ";", "product_description"
        )

        # Remove a newline character and lowercase words
        tables["advertised_products"] = self.transform_text(
            tables["advertised_products"], "product_description"
        )
        tables["product_keywords"] = self.transform_text(tables["product_keywords"], "keywords")

        # Merge product description and keywords dataframes
        tables["advertised_products"] = tables["advertised_products"].filter(
            regex="^(?!.*desc_).*$"
        )
        final_table = self.left_join_all(tables, "product_id")

        # Save preprocessed data to a shared work directory using ioctx
        output_path = self.ioctx.get_output_fn("data_preprocessing/preprocessed.parquet")
        final_table.to_parquet(output_path)

        self.logger.info("Preprocessing is done.")
