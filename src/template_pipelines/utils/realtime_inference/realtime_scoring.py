"""Script required for real time endpoint deployments to perform the inferencing/prediction."""

import logging
from pathlib import Path

import joblib

from template_pipelines.utils.realtime_inference.process_data import process_data


def init():
    """Init function required for real time scoring script.

    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    logging.info("Init script triggered.")

    model_dir_path = Path("/etc/models")
    model_path = next(path for path in model_dir_path.rglob("*.pkl") if path.suffix == ".pkl")
    model = joblib.load(str(model_path))
    logging.info("Init complete")


def run(raw_data):
    """This function is called for every invocation of the endpoint to perform the actual scoring/prediction."""
    return process_data(raw_data, model)
