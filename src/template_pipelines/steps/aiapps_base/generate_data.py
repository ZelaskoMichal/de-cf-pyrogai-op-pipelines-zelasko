"""GenerateDataStep class."""
import pandas as pd
from sklearn.datasets import make_classification

from aif.pyrogai.steps.step import Step


class GenerateDataStep(Step):
    """Generate Dataset to provider."""

    @staticmethod
    def generate_data(
        samples=1000,
        features=20,
        classes=2,
        informative=2,
        redundant=0,
        repeated=0,
        random_state=69,
        flip_y=0.1,
    ):
        """Generate synthetic data for testing machine learning algorithms."""
        X, y = make_classification(
            n_samples=samples,
            n_features=features,
            n_classes=classes,
            n_informative=informative,
            n_redundant=redundant,
            n_repeated=repeated,
            random_state=random_state,
            flip_y=flip_y,
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["target"] = y
        return df

    def run(self):
        """Runs step."""
        # Generate data
        df = None
        if "aiapps_sample.csv" in self.runtime_parameters.get("config_file_1"):
            df = pd.read_csv(self.inputs["config_file_1_input"])

        if "aiapps_sample.csv" in self.runtime_parameters.get("run_file_1"):
            df = pd.read_csv(self.inputs["run_file_1_input"])

        data = df if df is not None else self.generate_data(samples=200, features=10, informative=2)

        # Save output with iocontext
        fn = self.ioctx.get_output_fn("train_data.pkl")
        data.to_pickle(fn)

        self.logger.info("Data generated.")
