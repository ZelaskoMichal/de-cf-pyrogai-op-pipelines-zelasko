"""Unit tests for the EDAToolkit class."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from template_pipelines.utils.eda_pipeline.eda_toolkit import EDAToolkit


@pytest.fixture(scope="function")
def fixture_eda_toolkit():
    """Fixture for EDAToolkit."""
    data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4], "target": [10, 15]})
    logger = MagicMock()
    eda_toolkit = EDAToolkit(data, logger)
    return eda_toolkit


def test_generate_report_valid_tool(fixture_eda_toolkit):
    """Test generate_report with valid tools."""
    valid_tools = ["sweetviz", "klib", "dabl", "missingno"]

    for tool in valid_tools:
        with patch.object(EDAToolkit, f"_{tool}_report") as mock_method:
            fixture_eda_toolkit.generate_report(tool=tool)
            mock_method.assert_called_once()


def test_generate_report_invalid_tool(fixture_eda_toolkit):
    """Test generate_report with an invalid tool."""
    invalid_tool = "invalid_tool"
    fixture_eda_toolkit.generate_report(tool=invalid_tool)
    fixture_eda_toolkit.logger.warning.assert_called_once_with(
        f"Invalid tool selected: {invalid_tool}. Supported tools: {['sweetviz', 'klib', 'dabl', 'missingno']}"
    )


@patch("template_pipelines.utils.eda_pipeline.eda_toolkit.sv")
def test_sweetviz_report(mock_sv, fixture_eda_toolkit):
    """Test _sweetviz_report method."""
    fixture_eda_toolkit._sweetviz_report()
    mock_sv.analyze.assert_called_with(fixture_eda_toolkit.data)
    mock_sv.analyze.return_value.show_html.assert_called_with("sweetviz_report.html")
    fixture_eda_toolkit.logger.info.assert_called_with(
        "Sweetviz report generated: 'sweetviz_report.html'"
    )


@patch("template_pipelines.utils.eda_pipeline.eda_toolkit.klib")
def test_klib_report(mock_klib, fixture_eda_toolkit):
    """Test _klib_report method."""
    fixture_eda_toolkit._klib_report()
    fixture_eda_toolkit.logger.info.assert_any_call("Generating Klib report...")
    mock_klib.describe.assert_called_with(fixture_eda_toolkit.data)
    mock_klib.corr_plot.assert_called_with(fixture_eda_toolkit.data)
    mock_klib.missingno_matrix.assert_called_with(fixture_eda_toolkit.data)
    fixture_eda_toolkit.logger.info.assert_any_call("Klib report completed.")


@patch("template_pipelines.utils.eda_pipeline.eda_toolkit.plt")
@patch("template_pipelines.utils.eda_pipeline.eda_toolkit.dabl")
def test_dabl_report(mock_dabl, mock_plt, fixture_eda_toolkit):
    """Test _dabl_report method."""
    fixture_eda_toolkit._dabl_report()
    fixture_eda_toolkit.logger.info.assert_any_call("Generating Dabl report...")

    mock_dabl.plot.assert_called_once()
    fixture_eda_toolkit.logger.info.assert_any_call("Dabl report displayed.")


@patch("template_pipelines.utils.eda_pipeline.eda_toolkit.msno")
def test_missingno_report(mock_msno, fixture_eda_toolkit):
    """Test _missingno_report method."""
    fixture_eda_toolkit._missingno_report()
    fixture_eda_toolkit.logger.info.assert_any_call("Generating Missingno visualizations...")
    mock_msno.matrix.assert_called_with(fixture_eda_toolkit.data)
    mock_msno.bar.assert_called_with(fixture_eda_toolkit.data)
    fixture_eda_toolkit.logger.info.assert_any_call("Missingno visualizations displayed.")
