"""Unittests for dqge."""
from unittest.mock import patch

import pytest

from template_pipelines.steps.dq_ge.dq_ge_4_post_processing_data_validation import (
    PostProcessingDqStep,
)


@pytest.fixture(scope="function")
def fixture_post_processing_dq_step():
    """Fixture for PostProcessingDqStep."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        step = PostProcessingDqStep()
        yield step


def test_run(fixture_post_processing_dq_step):
    """Test run."""
    step = fixture_post_processing_dq_step

    with patch.object(step, "run_checks"), patch.object(
        step, "generate_data_docs"
    ) as mock_generate_data_docs, patch.object(step, "raise_for_status") as mock_raise_for_status:
        step.run()

        mock_generate_data_docs.assert_called_once()

        mock_raise_for_status.assert_called_once()
