"""Unit tests for DataValidationAfterFeatureCreationDqStep."""
from unittest.mock import MagicMock, patch

import pytest

from aif.pyrogai.const import Platform
from template_pipelines.steps.ml_training.dv_after_feature_creation import (
    DataValidationAfterFeatureCreationDqStep,
)


@pytest.fixture(scope="function")
def fixture_dv_step():
    """Fixture for DataValidationAfterFeatureCreationDqStep."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        dv_step = DataValidationAfterFeatureCreationDqStep()
        dv_step.outputs = {}
        yield dv_step


def test_run_method(fixture_dv_step):
    """Test the run method of DataValidationAfterFeatureCreationDqStep."""
    dv_step = fixture_dv_step

    dv_step.run_checks = MagicMock()
    dv_step.generate_data_docs = MagicMock()
    dv_step.raise_for_status = MagicMock()
    dv_step.ioctx.get_fns = MagicMock(return_value=iter(["fake_path/critical.html"]))
    dv_step.platform = Platform.AML

    dv_step.run()

    dv_step.run_checks.assert_called_once()
    dv_step.generate_data_docs.assert_called_once()
    dv_step.raise_for_status.assert_called_once()
    dv_step.ioctx.get_fns.assert_called_once_with(
        "ge_dq_datadocs/{}/**/critical.html".format(dv_step.step_name)
    )

    assert dv_step.outputs["mlflow_ge_doc"] == "fake_path/critical.html"
