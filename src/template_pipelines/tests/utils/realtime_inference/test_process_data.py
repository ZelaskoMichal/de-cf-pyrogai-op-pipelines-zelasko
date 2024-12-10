"""Unittests."""

import json
from unittest.mock import Mock

import numpy as np

from template_pipelines.utils.realtime_inference.process_data import process_data


def test_process_data():
    """Test process data."""
    raw_data = json.dumps(
        {
            "data": [
                {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                {"sepal_length": 4.9, "sepal_width": 3.0, "petal_length": 1.4, "petal_width": 0.2},
            ]
        }
    )

    expected_output = [0, 1]

    model = Mock()
    model.predict.return_value = np.array(expected_output)

    result = process_data(raw_data, model)
    assert result == expected_output

    expected_call_arg = np.array([[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]])
    np.testing.assert_array_equal(model.predict.call_args[0][0], expected_call_arg)
