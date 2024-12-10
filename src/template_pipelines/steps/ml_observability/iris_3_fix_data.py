"""FixDataStep class."""
import pandas as pd

from aif.pyrogai.steps.step import Step


class FixDataStep(Step):
    """Fixes known errors in the Iris dataset.

    The 35th sample should be: 4.9,3.1,1.5,0.2,"Iris-setosa"
    where the error is in the fourth feature.

    The 38th sample should be: 4.9,3.6,1.4,0.1,"Iris-setosa"
    where the errors are in the second and third feature.

    source: https://archive.ics.uci.edu/ml/datasets/Iris
    """

    def run(self):
        """Runs step."""
        # load input with iocontext
        fn = self.ioctx.get_fn("standarised.csv")
        data = pd.read_csv(fn)
        self.logger.info(f"Input row count {len(data)}")

        # fix data (rows are indexed from 0)
        data.loc[34, "petal_width"] = 0.2
        data.loc[37, "sepal_width"] = 3.6
        data.loc[37, "petal_length"] = 1.4

        # save output with iocontext
        fn = self.ioctx.get_output_fn("fixed.csv")
        data.to_csv(fn)

        self.logger.info("Iris data fixed.")
