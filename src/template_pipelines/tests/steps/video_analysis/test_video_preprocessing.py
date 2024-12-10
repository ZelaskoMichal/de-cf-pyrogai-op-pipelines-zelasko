"""Tests for video processing."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from aif.pyrogai.steps.step import Step
from template_pipelines.steps.video_analysis.video_processing import VideoProcessing


class MockIoctx:
    """A mock class for Pyrogai ioctx."""

    def __init__(self, value):
        """Initialize a MockIoctx instance."""
        self.value = value

    def get_output_fn(self, arg):
        """Return the stored value."""
        return self.value


class MockBaseGenAIStep(Step):
    """A mock BaseGenAIStep for testing purposes."""

    def __init__(self):
        """Initialize a MockBaseGenAIStep instance."""
        super().__init__()

    def mock_super_init(self):
        """Mock initialization for setting up logging, secrets and configuration."""
        self.logger = logging
        self.secrets = {}
        self.config = {"video_analysis": {"thread_num": 1}}


@pytest.fixture(scope="function")
def fixture_video_processing():
    """Fixture for the VideoProcessing step."""
    with (
        patch(
            "template_pipelines.steps.video_analysis.video_processing.BaseGenAIStep.__init__",
            MockBaseGenAIStep.mock_super_init,
        ),
        patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"),
    ):
        vp = VideoProcessing()
        yield vp


@patch("template_pipelines.steps.video_analysis.video_processing.VideoSummaryPrompt.prompt")
@patch.object(VideoProcessing, "ask_gemini")
@patch("template_pipelines.steps.video_analysis.video_processing.run_multithreaded")
@patch("template_pipelines.steps.video_analysis.video_processing.pd.read_csv")
def test_video_processing_run(
    mock_pd_read_csv, mock_run_multithreaded, mock_ask_gemini, mock_prompt, fixture_video_processing
):
    """Test the VideoProcessing run method."""
    mock_pd_read_csv.return_value = None
    mock_prompt.return_value = "This is a test"
    fixture_video_processing.inputs = MagicMock()
    fixture_video_processing.ioctx = MockIoctx("output.txt")
    fixture_video_processing.run()

    mock_pd_read_csv.assert_called_once()
    mock_run_multithreaded.assert_called_once_with(
        mock_ask_gemini, (mock_prompt, None), 1, "output.txt"
    )
