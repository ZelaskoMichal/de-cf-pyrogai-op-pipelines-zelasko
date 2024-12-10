"""Module for the  ModelInferencer class."""

import logging
import typing as t

import joblib
from iiotpy.adapter import AdapterBase
from iiotpy.scaffolding import DataSetRequestRoute

from .scaffolding import IrisObject, convert_zmq_time

logger = logging.getLogger()


class ModelInferencer:  # pylint: disable=too-few-public-methods
    """Class that handles model inferencing.

    Processes the received input data, generates the prediction and posts the result.
    """

    _current_iris_object = IrisObject
    _adapter = AdapterBase
    _model = t.Any

    def __init__(self, adapter: AdapterBase) -> None:
        """Initializes the ModelInferencer class."""
        self._current_iris_object = IrisObject()
        self._adapter = adapter

        with open("models/mdf_model/model.pkl", "rb") as fid:
            self._model = joblib.load(fid)

    def process_message(self, message: t.Dict[str, t.Any]) -> None:
        """Function used to process received iris messages."""
        for item in message["Items"]:
            logger.debug("Processing message item: %s", item)

            if not self._current_iris_object.assign_value_from_message_item(item):
                continue

            if self._current_iris_object.is_complete:
                prediction = self._predict(self._current_iris_object)
                logger.debug("Prediction is: %s", prediction)
                self._post_response_message(prediction)
            else:
                logger.debug("Received incomplete petal data, discarding")

            logger.debug("created new Iris Object")
            self._current_iris_object = IrisObject()

    def _predict(self, value: IrisObject) -> int:
        """Runs the ML model against the value passed in."""
        input_data = [
            [value.sepal_length, value.sepal_width, value.petal_length, value.petal_width]
        ]
        prediction = self._model.predict(input_data)

        return prediction[0]

    def _post_response_message(self, prediction: int) -> None:
        """Sends the prediction and data to the upstream storage manager."""
        try:
            zmq_time = convert_zmq_time(self._current_iris_object.time_stamp)
        except Exception:  # pylint: disable=broad-exception-caught
            self._adapter.request_shutdown()
            return

        artifacts = self._current_iris_object.to_dictionary()
        artifacts["irisType"] = prediction

        self._adapter.databus.post_data_set_request(
            data_path_key="irisResult",
            artifacts=artifacts,
            route=DataSetRequestRoute.ML2STORAGEMANAGER,
            timestamp=zmq_time,
        )
