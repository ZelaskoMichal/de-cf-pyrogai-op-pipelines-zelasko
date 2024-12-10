"""Tests for toolkit."""

from concurrent.futures import Future
from dataclasses import is_dataclass
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd

from template_pipelines.utils.video_analysis import toolkit


@patch("template_pipelines.utils.video_analysis.toolkit.open", new_callable=mock_open)
@patch("template_pipelines.utils.video_analysis.toolkit.as_completed")
@patch("template_pipelines.utils.video_analysis.toolkit.ThreadPoolExecutor")
def test_run_multithreaded(mock_executor, mock_as_completed, mock_file):
    """Test run_multithreaded."""
    query = "process"
    func = lambda query, row: (row["id"], f"{query}ed_{row['id']}")
    df = pd.DataFrame({"id": [1, 2]})
    output_path = "test_output.txt"

    mock_future_1 = MagicMock(spec=Future)
    mock_future_1.result.return_value = func(query, {"id": 1})

    mock_future_2 = MagicMock(spec=Future)
    mock_future_2.result.return_value = func(query, {"id": 2})

    mock_as_completed.return_value = [mock_future_1, mock_future_2]

    toolkit.run_multithreaded(func, (query, df), 2, output_path)

    assert mock_executor.return_value.__enter__.return_value.submit.call_count == 2
    mock_file().write.assert_any_call("1; processed_1\n")
    mock_file().write.assert_any_call("2; processed_2\n")


def test_video_summary_prompt():
    """Test VideoSummaryPrompt."""
    assert is_dataclass(toolkit.VideoSummaryPrompt)
    assert isinstance(toolkit.VideoSummaryPrompt.prompt, str)
