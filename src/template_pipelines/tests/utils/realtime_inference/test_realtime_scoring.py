"""Unittests."""
from unittest.mock import MagicMock, patch

from src.template_pipelines.utils.realtime_inference.realtime_scoring import init


@patch("src.template_pipelines.utils.realtime_inference.realtime_scoring.Path.rglob")
@patch("src.template_pipelines.utils.realtime_inference.realtime_scoring.joblib.load")
def test_init(mock_load, mock_rglob):
    """Test init."""
    mock_model = MagicMock()
    mock_load.return_value = mock_model
    mock_path = MagicMock()
    mock_path.suffix = ".pkl"
    mock_rglob.return_value = [mock_path]

    init()

    mock_rglob.assert_called_once_with("*.pkl")
    mock_load.assert_called_once_with(str(mock_path))
