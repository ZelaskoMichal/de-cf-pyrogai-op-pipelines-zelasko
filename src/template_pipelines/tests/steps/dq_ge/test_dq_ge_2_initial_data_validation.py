"""Unittests for dqge."""
from unittest.mock import patch

import pytest

from template_pipelines.steps.dq_ge.dq_ge_2_initial_data_validation import InitialDqStep


@pytest.fixture(scope="function")
def fixture_initial_dq():
    """Fixture for InitialDqStep."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        ids = InitialDqStep()
        yield ids


def test_run(fixture_initial_dq):
    """Test run."""
    ids = fixture_initial_dq

    with patch.object(ids, "run_checks") as mock_run_checks, patch.object(
        ids, "generate_data_docs"
    ) as mock_generate_data_docs, patch.object(ids, "raise_for_status") as mock_raise_for_status:
        ids.run()

        # Check the logger calls
        ids.logger.info.assert_any_call("Start initial dq after data loading step")
        ids.logger.info.assert_any_call("Finish initial dq after data loading step")

        # Check the method calls
        mock_run_checks.assert_called_once_with(step_name=ids.step_name)
        mock_generate_data_docs.assert_called_once()
        mock_raise_for_status.assert_called_once()
