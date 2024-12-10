"""ScoreDataStep class."""
import pickle

import pandas as pd

from aif.pyrogai.steps.step import Step
from template_pipelines.utils.aiapps_base.io_helpers import write_csv_file, write_json_file
from template_pipelines.utils.aiapps_base.score_data_step_helpers import (
    generate_results_table,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    score_data,
)


class ScoreDataStep(Step):
    """Scores data."""

    def run(self):
        """Runs step."""
        # load input with iocontext
        fn_trained_model = self.ioctx.get_fn("trained_model.pickle")
        with open(fn_trained_model, "rb") as f:
            model = pickle.load(f)

        data = self.ioctx.get_fn("preprocessed_data.csv")
        data_df = pd.read_csv(data)
        x_train = data_df.drop(columns=["target"])
        y_train = data_df["target"]

        # Calculating train scores and model results
        train_predictions, train_accuracy = score_data(model, x_train, y_train)

        # Generate visualizations as data objects
        confusion_matrix_obj = plot_confusion_matrix(y_train, train_predictions)
        precision_recall_curve_obj = plot_precision_recall_curve(
            y_train, model.predict_proba(x_train)[:, 1]
        )
        roc_curve_obj = plot_roc_curve(y_train, model.predict_proba(x_train)[:, 1])

        results_table = generate_results_table(train_predictions, y_train)

        fn_model_results = self.ioctx.get_output_fn("model_results.json")
        with open(fn_model_results, "w") as f:
            results_table.to_json(fn_model_results, orient="records")

        self.logger.info(f"Train Accuracy: {train_accuracy}")
        self.logger.info("Data scored.")

        # AI Apps Insights - Generate data for plotting
        data_dict = {"data": []}

        test = results_table["Actual"].value_counts().reset_index()
        predictions = results_table["Predicted"].value_counts().reset_index()

        actual_obj = {
            "type": "bar",
            "name": "actual",
            "x": test["index"].tolist(),
            "y": test["Actual"].tolist(),
        }

        predicted_obj = {
            "type": "bar",
            "name": "predicted",
            "x": predictions["index"].tolist(),
            "y": predictions["Predicted"].tolist(),
        }

        data_dict["data"].append(actual_obj)
        data_dict["data"].append(predicted_obj)

        layaout = {
            "width": 1200,
            "height": 500,
            "title": "Model Predicted vs Actual",
        }

        data_dict["layout"] = layaout.copy()

        # Options for dropdown
        graph_options = {
            "label": "Results Type",
            "options": [
                {"name": "Model Predicted vs Actual", "path": "validation_predictions.json"},
                {"name": "Confusion Matrix", "path": "confusion_matrix.json"},
                {"name": "Precision Recall Curve", "path": "precision_recall_curve.json"},
                {"name": "ROC Curve", "path": "roc_curve.json"},
            ],
        }

        self.logger.info(f"Provider: {self.provider}")
        self.logger.info(f"Config Module: {self.config_module}")

        # Write json file for chart data options
        json_file_name = self.config["aiapps_outputs_files"]["file_1"]["name"]

        self.outputs["results_options_json"] = write_json_file(graph_options, json_file_name)

        # Write json file for confusion matrix
        json_file_name = self.config["aiapps_outputs_files"]["file_2"]["name"]
        layaout["title"] = "Confusion Matrix"
        new_data = {"data": [confusion_matrix_obj], "layout": layaout}

        self.outputs["confusion_matrix"] = write_json_file(new_data, json_file_name)

        # Write json file for precision recall curve
        json_file_name = self.config["aiapps_outputs_files"]["file_3"]["name"]
        layaout["title"] = "Precision Recall Curve"
        new_data = {"data": [precision_recall_curve_obj], "layout": layaout}

        self.outputs["precision_recall_curve"] = write_json_file(new_data, json_file_name)

        # Write json file for roc curve
        json_file_name = self.config["aiapps_outputs_files"]["file_4"]["name"]
        layaout["title"] = "ROC Curve"
        new_data = {"data": [roc_curve_obj], "layout": layaout}

        self.outputs["roc_curve"] = write_json_file(new_data, json_file_name)

        # Write csv file for table data
        csv_file_name = self.config["aiapps_outputs_files"]["file_5"]["name"]

        self.outputs["validation_predictions_csv"] = write_csv_file(results_table, csv_file_name)

        # Write json file for validation predictions bar
        json_file_name = self.config["aiapps_outputs_files"]["file_6"]["name"]

        self.outputs["validation_predictions_json"] = write_json_file(data_dict, json_file_name)

        # Print params
        for param_name, param_value in self.runtime_parameters.items():
            self.logger.info(f"Param name: {param_name}, param value: {param_value}")

        # Write files under uploads directory to show file referencing within step

        if self.runtime_parameters.get("config_file_1") not in ["None", None, ""]:
            file_path = self.inputs["config_file_1_input"]

            with open(file_path) as fp:
                fp.read()
                self.outputs["config_file_1_output"] = fp.name

        if self.runtime_parameters.get("run_file_1") not in ["None", None, ""]:
            file_path = self.inputs["run_file_1_input"]

            with open(file_path) as fp:
                fp.read()
                self.outputs["run_file_1_output"] = fp.name
