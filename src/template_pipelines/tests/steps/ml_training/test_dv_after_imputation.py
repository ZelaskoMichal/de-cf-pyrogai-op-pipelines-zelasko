"""Unit tests for DataValidationAfterFeatureCreationDqStep."""
from unittest.mock import MagicMock, patch

import pytest

from aif.pyrogai.const import Platform
from template_pipelines.steps.ml_training.dv_after_imputation import (
    DataValidationAfterImputationDqStep,
)


@pytest.fixture(scope="function")
def fixture_dv_imputation_step():
    """Fixture for DataValidationAfterImputationDqStep."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        dv_step = DataValidationAfterImputationDqStep()
        dv_step.outputs = {}
        yield dv_step


def test_run_method(fixture_dv_imputation_step):
    """Test the run method of DataValidationAfterImputationDqStep."""
    dv_step = fixture_dv_imputation_step

    dv_step.run_checks = MagicMock()
    dv_step.generate_data_docs = MagicMock()
    dv_step.raise_for_status = MagicMock()

    dv_step.ioctx.get_fns = MagicMock(return_value=iter(["fake_path/warning.html"]))
    dv_step.platform = Platform.AML
    dv_step.run()
    assert dv_step.outputs["mlflow_ge_doc"] == "fake_path/warning.html"

    dv_step.outputs.clear()

    dv_step.ioctx.get_fns = MagicMock(return_value=iter(["fake_path/warning.html"]))
    dv_step.platform = Platform.VERTEX
    with patch("builtins.open", MagicMock()) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = "fake_html_content"
        dv_step.run()
        assert dv_step.outputs["kfp_ge_doc"] == "fake_html_content"
