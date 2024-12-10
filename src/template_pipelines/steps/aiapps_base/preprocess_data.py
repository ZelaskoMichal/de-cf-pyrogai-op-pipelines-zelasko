"""PreprocessDataStep class."""
import pandas as pd
from sklearn.preprocessing import StandardScaler

from aif.pyrogai.steps.step import Step


class PreprocessDataStep(Step):
    """Preprocesses dataset."""

    @staticmethod
    def preprocess_data(X):
        """Preprocess the input data."""
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(X)
        return x_scaled

    def run(self):
        """Runs step."""
        # load input with iocontext
        fn = self.ioctx.get_fn("train_data.pkl")
        data_df = pd.read_pickle(fn)
        self.logger.info(f"Input row count {len(data_df)}")

        x_train = data_df.drop(columns=["target"])
        y_train = data_df["target"]
        x_train_scaled = self.preprocess_data(x_train)
        df_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns)
        df_scaled["target"] = y_train

        # save output with iocontext
        fn = self.ioctx.get_output_fn("preprocessed_data.csv")
        df_scaled.to_csv(fn)

        self.logger.info("Data standardized and saved.")
