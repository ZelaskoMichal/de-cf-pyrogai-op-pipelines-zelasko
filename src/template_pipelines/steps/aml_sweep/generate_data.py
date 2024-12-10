"""Step for generating data."""

import pickle
import random
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from aif.pyrogai.steps import Step


class GenerateData(Step):
    """Creates some split data and saves it to ioctx."""

    def run(self) -> None:
        """User step logic."""
        df = self._generate_dataframe()
        train, test, enc = self._process_data(df)
        # save data to ioctx
        fn = self.ioctx.get_output_fn("data/train.parquet")
        train.to_parquet(fn)
        fn = self.ioctx.get_output_fn("data/test.parquet")
        test.to_parquet(fn)
        # save encoder for later use?
        fn = self.ioctx.get_output_fn("encoder.pkl")
        with open(fn, "wb") as f:
            pickle.dump(enc, f)

    @staticmethod
    def _generate_dataframe(num_rows=500) -> pd.DataFrame:
        """Generate test data for simplicity."""
        col_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
        data = []
        for _ in range(num_rows):
            values = [random.uniform(4.0, 7.9) for _ in range(4)]
            class_label = "Iris-" + random.choice(["setosa", "versicolor", "virginica"])
            values.append(class_label)  # type: ignore
            data.append(values)
        return pd.DataFrame(data, columns=col_names)

    @staticmethod
    def _process_data(df) -> Tuple[pd.DataFrame, pd.DataFrame, LabelEncoder]:
        """Process data for sweep step."""
        enc = LabelEncoder()
        df["species"] = enc.fit_transform(df["species"])
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        return train, test, enc
