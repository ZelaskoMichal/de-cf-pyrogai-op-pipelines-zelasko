"""Powered by the IIOT Adapter Framework!"""

import logging
import typing as t

from iiotpy.adapter import Adapter
from infrastructure.model_inferencer import ModelInferencer

#: Adapter is the main application and must be initialized first
app = Adapter()

#: Init our logger
logger = logging.getLogger()

data_processor = ModelInferencer(app)


@app.handle_time_series_message("iris-measurements")
def handle_message(message: t.Dict[str, t.Any]) -> None:
    """Function that handles time series messages."""
    logger.info("Handling TimeSeriesRequest from ML Manager")
    data_processor.process_message(message)


if __name__ == "__main__":
    app.run()
