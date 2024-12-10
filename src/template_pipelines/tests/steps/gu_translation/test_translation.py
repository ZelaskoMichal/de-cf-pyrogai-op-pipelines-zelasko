"""Tests for GU Translation."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aif.pyrogai.steps.step import Step
from template_pipelines.steps.gu_translation.translation import Translation


class MockBaseGenAIStep(Step):
    """A mock BaseGenAIStep for testing purposes."""

    def __init__(self):
        """Initialize a MockBaseGenAIStep instance."""
        super().__init__()

    def mock_super_init(self):
        """Mock initialization for setting up logging and secrets."""
        self.logger = MagicMock()
        self.secrets = {}


@pytest.fixture(scope="function")
def fixture_translation():
    """Fixture for the Translation step."""
    with (
        patch(
            "template_pipelines.steps.gu_translation.translation.BaseGenAIStep.__init__",
            MockBaseGenAIStep.mock_super_init,
        ),
        patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"),
    ):
        translation = Translation()
        yield translation


def test_run_multithreaded(fixture_translation):
    """Test run_multithreaded."""
    func = lambda text, original, target: f"{text}_{original}_{target}"
    func_inputs = ("text", "original", ["target1", "target2"])

    df = fixture_translation.run_multithreaded(func, func_inputs, 2)

    expected_df = pd.DataFrame(
        {
            "language": ["original", "target1", "target2"],
            "translation": ["text", "text_original_target1", "text_original_target2"],
        }
    )
    expected_df = expected_df.sort_values(by="language").reset_index(drop=True)
    df = df.sort_values(by="language").reset_index(drop=True)

    assert isinstance(df, pd.DataFrame)
    pd.testing.assert_frame_equal(df, expected_df, check_like=True)


@patch.object(Translation, "run_multithreaded")
@patch.object(Translation, "translate")
def test_translation_run(mock_translate, mock_run_multithreaded, fixture_translation):
    """Text the Translation run method."""
    mock_translate.side_effect = lambda text, original, target: f"{text}_{original}_{target}"
    expected_df = pd.DataFrame(
        {
            "language": ["original", "target1", "target2"],
            "translation": ["text", "text_original_target1", "text_original_target2"],
        }
    )
    mock_run_multithreaded.return_value = expected_df
    fixture_translation.runtime_parameters = {
        "text": "text",
        "original_language": "original",
        "target_languages": "target1,target2",
    }
    fixture_translation.config = {"gu_translation": {"thread_num": 2}}

    fixture_translation.run()

    mock_run_multithreaded.assert_called_once_with(
        fixture_translation.translate, ("text", "original", ["target1", "target2"]), 2
    )
    fixture_translation.logger.info.assert_any_call("Running GU translation...")
    fixture_translation.logger.info.assert_any_call(
        f"The text: 'text' has been translated into following languages:\n{expected_df}"
    )
    fixture_translation.logger.info.assert_any_call("Translation is done.")
