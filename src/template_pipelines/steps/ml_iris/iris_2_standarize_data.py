"""StandardizeDataStep class."""
import pandas as pd

from aif.pyrogai.steps.step import Step


class StandardizeDataStep(Step):
    """Standardizes dataset.

    Standardization includes:
    - make all column names lowercase
    - set proper data types for feature columns
    - remove dashes from target column
    """

    def run(self):
        """Runs step."""
        # load input with iocontext
        fn = self.ioctx.get_fn("uploaded_data.parquet")
        data = pd.read_parquet(fn)
        self.logger.info(f"Input row count {len(data)}")

        # define local variables names for convenience
        target = self.config["ml_iris"]["target"]
        features = self.config["ml_iris"]["features"]

        # standardize column names - make them lowercase
        data.columns = [x.lower() for x in data.columns]

        # set proper data types for features columns
        for column_name in data.columns:
            if column_name in features:
                data[column_name] = data[column_name].astype("double")

        # standardize target column values - changes dashes to spaces
        data[target] = data[target].apply(lambda x: x.replace("-", " "))

        # save output with iocontext
        fn = self.ioctx.get_output_fn("standarised.csv")
        data.to_csv(fn)

        self.logger.info("Iris data standardized and saved.")
