"""SplitDataStep class."""
import pandas as pd
from sklearn.model_selection import train_test_split

from aif.pyrogai.steps.step import Step


class SplitDataStep(Step):
    """Splits data into train and test parts.

    Parameters:
    - stratify_columns - list of columns to stratify dataset on (default: None),
    - random_state - random seed to split data (default: None)
    - train_size - number / percentage of rows to include in train part (default: 0.8)
    """

    def run(self):
        """Runs step."""
        # load input with iocontext
        fn = self.ioctx.get_fn("fixed.csv")
        data = pd.read_csv(fn)
        self.logger.info(f"Input row count {len(data)}")

        # check configuration
        stratify_column = self.config["ml_observability"]["target"]
        random_state = self.runtime_parameters.get("random_state", None)
        train_size = self.runtime_parameters.get("train_size", 0.8)

        # split data
        if stratify_column:
            train_data, test_data = train_test_split(
                data,
                train_size=train_size,
                stratify=data[stratify_column],
                random_state=random_state,
            )
        else:
            train_data, test_data = train_test_split(
                data, train_size=train_size, random_state=random_state
            )

        # save output with iocontext
        fn_train_data = self.ioctx.get_output_fn("train_data.parquet")
        train_data.to_parquet(fn_train_data)

        fn_test_data = self.ioctx.get_output_fn("test_data.parquet")
        test_data.to_parquet(fn_test_data)

        self.logger.info("Iris data splitted.")
