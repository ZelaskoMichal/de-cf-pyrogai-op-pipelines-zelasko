"""ScoreData Step helpers."""
import pandas as pd
from sklearn.metrics import accuracy_score, auc, confusion_matrix, precision_recall_curve, roc_curve


def score_data(model, X, y):
    """Score the given data using the provided model."""
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    return predictions, accuracy


def generate_results_table(predictions, y):
    """Generate a results table based on predicted and actual."""
    results_table = pd.DataFrame({"Predicted": predictions, "Actual": y})
    results_table["Validation"] = results_table["Predicted"] == results_table["Actual"]
    return results_table


def plot_confusion_matrix(y_true, y_pred):
    """Plot the confusion matrix for the predicted and true labels."""
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "heatmap",
        "name": "Confusion Matrix",
        "x": ["Predicted Negative", "Predicted Positive"],
        "y": ["Actual Negative", "Actual Positive"],
        "z": cm.tolist(),
    }


def plot_precision_recall_curve(y_true, y_score):
    """Plot the precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return {
        "type": "scatter",
        "name": "Precision-Recall Curve",
        "x": recall.tolist(),
        "y": precision.tolist(),
    }


def plot_roc_curve(y_true, y_score):
    """Plot the Receiver Operating Characteristic (ROC) curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return {
        "type": "scatter",
        "name": "ROC Curve (AUC = {:.2f})".format(roc_auc),
        "x": fpr.tolist(),
        "y": tpr.tolist(),
    }
