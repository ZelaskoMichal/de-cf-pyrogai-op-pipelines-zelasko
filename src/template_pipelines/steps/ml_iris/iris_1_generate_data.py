"""GenerateDataStep class."""
import random

import pandas as pd

from aif.pyrogai.steps.step import Step


class GenerateDataStep(Step):
    """Generate Iris Dataset to provider."""

    def generate_dataframe(self, num_rows=150):
        """Generate random dataframe with Iris Dataset."""
        data = []
        for _ in range(num_rows):
            values = [random.uniform(4.0, 7.9) for _ in range(4)]
            class_label = "Iris-" + random.choice(["setosa", "versicolor", "virginica"])
            values.append(class_label)
            data.append(values)

        columns = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "class",
        ]
        df = pd.DataFrame(data, columns=columns)
        return df

    def run(self):
        """Runs step."""
        # Generate data
        data = self.generate_dataframe()

        # Save output with iocontext
        fn = self.ioctx.get_output_fn("uploaded_data.parquet")
        data.to_parquet(fn)

        self.logger.info("Iris data generated.")
