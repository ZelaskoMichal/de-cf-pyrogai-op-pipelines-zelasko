"""Unit tests for template_pipelines/utils/aml_sweep/io_utils.py."""

from unittest import mock

from aif.pyrogai.const import Platform
from template_pipelines.utils.aml_sweep.io_utils import get_ioslot_name


def test_get_ioslot_name():
    """Tests of get_ioslot_name function."""
    base_ioslot_name = "base_nAme"
    step = mock
    step.platform = Platform.AML
    assert get_ioslot_name(step, base_ioslot_name) == "base_nAme"
    step.platform = "not AML"
    assert get_ioslot_name(step, base_ioslot_name) == "cf_base_nAme"
