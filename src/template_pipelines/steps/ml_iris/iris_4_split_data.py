"""SplitDataStep class."""
import pandas as pd

from aif.pyrogai.steps.step import Step


class SplitDataStep(Step):
    """Splits data into train and test parts.

    Parameters:
    - stratify_columns - list of columns to stratify dataset on (default: None),
    - random_state - random seed to split data (default: None)
    - train_size - number / percentage of rows to include in train part (default: 0.8)
    """

    def custom_train_test_split(self, *args, **kwargs):
        """Custom train_test_split method."""
        example_data = {
            "sepal_length": [6.7, 5.8, 6.7, 7.6, 4.5, 5.0, 6.0, 5.5, 6.3, 5.5],
            "sepal_width": [3.0, 4.0, 2.5, 3.0, 2.3, 3.4, 2.2, 2.4, 2.5, 3.5],
            "petal_length": [5.0, 1.2, 5.8, 6.6, 1.3, 1.5, 5.0, 3.7, 4.9, 1.3],
            "petal_width": [1.7, 0.2, 1.8, 2.1, 0.3, 0.2, 1.5, 1.0, 1.5, 0.2],
            "class": [
                "Iris versicolor",
                "Iris setosa",
                "Iris virginica",
                "Iris virginica",
                "Iris setosa",
                "Iris setosa",
                "Iris virginica",
                "Iris versicolor",
                "Iris versicolor",
                "Iris setosa",
            ],
        }
        return pd.DataFrame(example_data), pd.DataFrame(example_data)

    def run(self):
        """Runs step."""
        # load input with iocontext
        fn = self.ioctx.get_fn("fixed.csv")
        data = pd.read_csv(fn)
        self.logger.info(f"Input row count {len(data)}")

        # check configuration
        stratify_columns = [self.config["ml_iris"]["target"]]
        random_state = self.runtime_parameters.get("random_state", None)
        train_size = self.runtime_parameters.get("train_size", 0.8)

        # split data
        if stratify_columns:
            train_data, test_data = self.custom_train_test_split(
                data,
                train_size=train_size,
                random_state=random_state,
                stratify=data[stratify_columns],
            )
        else:
            train_data, test_data = self.custom_train_test_split(
                data, train_size=train_size, random_state=random_state
            )

        # save output with iocontext
        fn_train_data = self.ioctx.get_output_fn("train_data.parquet")
        train_data.to_parquet(fn_train_data)

        fn_test_data = self.ioctx.get_output_fn("test_data.parquet")
        test_data.to_parquet(fn_test_data)

        self.logger.info("Iris data splitted.")
