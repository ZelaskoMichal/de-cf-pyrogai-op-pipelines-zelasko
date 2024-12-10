"""Eda example class."""
import pandas as pd
from sklearn import datasets

from aif.pyrogai.steps.step import Step
from template_pipelines.utils.eda_pipeline.eda_toolkit import EDAToolkit


class EdaExample(Step):
    """Perform EDA on Iris Dataset loaded from a library."""

    def load_iris_data(self):
        """Load Iris dataset from sklearn and convert to pandas DataFrame."""
        self.logger.info("Loading Iris dataset...")
        iris = datasets.load_iris()
        data = pd.DataFrame(iris.data, columns=iris.feature_names)
        data["target"] = iris.target
        return data

    def perform_eda(self, data, tool="pandas_profiling"):
        """Perform EDA using the specified tool."""
        eda_toolkit = EDAToolkit(data, self.logger)
        eda_toolkit.generate_report(tool)

    def run(self):
        """Runs step."""
        # Step 1: Load the Iris dataset
        data = self.load_iris_data()

        # Step 2: Perform EDA
        self.logger.info("Performing EDA...")
        selected_tool = self.config.get("eda_tool")
        self.perform_eda(data, tool=selected_tool)

        self.logger.info("EDA step completed.")
