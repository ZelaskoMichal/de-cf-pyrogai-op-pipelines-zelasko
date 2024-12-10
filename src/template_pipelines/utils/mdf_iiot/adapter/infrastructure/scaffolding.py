"""Module that contains helper classes and functions."""

import logging
import typing as t
from datetime import datetime

import pandas as pd

logger = logging.getLogger()


class IrisObject:
    """Object that handles parsing iris message data.

    Detects when all the data for a given flower has been received.
    """

    sepal_length = float
    sepal_width = float
    petal_length = float
    petal_width = float
    time_stamp = datetime

    def __init__(self) -> None:
        """Initializes the IrisObject class."""
        self.sepal_length = 0
        self.sepal_width = 0
        self.petal_length = 0
        self.petal_width = 0
        self.time_stamp = None

    @property
    def is_complete(self) -> bool:
        """Gets a boolean value indicating if every feature has a value greater than zero."""
        return (
            self.sepal_length > 0
            and self.sepal_width > 0
            and self.petal_length > 0
            and self.petal_width > 0
        )

    def assign_value_from_message_item(self, item) -> bool:
        """Parses the incoming message and assigns it to the correct property value.

        The function will return True if the item collected is the Petal Width,
        otherwise the return value will be False.
        """
        return_value = False

        if item["Tag"] == "Iris.SepalLength":
            self.sepal_length = item["NumericValue"]
            logger.debug("Init Sepal Length with %s", item["NumericValue"])
        elif item["Tag"] == "Iris.SepalWidth":
            self.sepal_width = item["NumericValue"]
            logger.debug("Init Sepal Width with %s", item["NumericValue"])
        elif item["Tag"] == "Iris.PetalLength":
            self.petal_length = item["NumericValue"]
            logger.debug("Init Petal Length with %s", item["NumericValue"])
        elif item["Tag"] == "Iris.PetalWidth":
            self.petal_width = item["NumericValue"]
            logger.debug("Init Petal Width with %s", item["NumericValue"])
            return_value = True
        else:
            return False

        self.time_stamp = item["TimeStamp"]
        return return_value

    def to_dictionary(self) -> dict[str, t.Any]:
        """Converts the current IrisObject instance into a dictionary."""
        return {
            "Iris.SepalLength": self.sepal_length,
            "Iris.SepalWidth": self.sepal_width,
            "Iris.PetalLength": self.petal_length,
            "Iris.PetalWidth": self.petal_width,
        }


def convert_zmq_time(time: str) -> datetime:
    """Function used to convert time in the ZMQ format into a standard timestamp."""
    # Ensure correct timestamp format
    try:
        # Convert string to datetime
        time = pd.Timestamp(datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ"))
    except Exception:  # pylint: disable=broad-exception-caught
        try:
            # modify datetime format to include microseconds
            time = time[:-1] + ".000000Z"
            time = pd.Timestamp(datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ"))
        except Exception:
            logging.exception("Got exception while converting ZMQ Time")
            raise

    return time
