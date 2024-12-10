"""LogModel class."""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from aif.pyrogai.steps.step import Step  # noqa: E402


class LogModel(Step):
    """Logging model step."""

    def run(self):
        """Run step."""
        data = load_iris()
        x, y = data.data, data.target

        # Split the data
        x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42)

        # Create and train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)

        model_info = self.mlflow.log_model_with_tags(
            classifier=model,
            model_name=self.config["model_name"],
            flavor=self.config["model_flavor"],
            tags={"model_type": self.config["model_type"]},
        )
        self.outputs["mdf_model_uri"] = model_info.model_uri
