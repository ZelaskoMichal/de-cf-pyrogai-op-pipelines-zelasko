"""File that checks if real-time inference can correctly access user project code."""

import json
import logging

import numpy as np


def process_data(raw_data, model):
    """Extract the "data" field from a payload sent to a real time endpoint."""
    data = json.loads(raw_data)["data"]
    data = np.array(
        [[d["sepal_length"], d["sepal_width"], d["petal_length"], d["petal_width"]] for d in data]
    )
    result = model.predict(data)

    logging.info("Request processed")

    return result.tolist()
