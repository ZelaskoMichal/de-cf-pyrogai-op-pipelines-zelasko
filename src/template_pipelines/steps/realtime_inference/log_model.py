"""Preprocessing step class."""


from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from aif.pyrogai.steps.step import Step  # noqa: E402


class LogModel(Step):
    """Preprocessing step."""

    def run(self):
        """Run preprocessing step."""
        data = load_iris()
        x, y = data.data, data.target

        # Split the data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Create and train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)

        model_info = self.mlflow.sklearn.log_model(
            model, "realtime_inference_model", registered_model_name="realtime_inference_model"
        )
        self.outputs["realtime_inference_model"] = model_info.model_uri
