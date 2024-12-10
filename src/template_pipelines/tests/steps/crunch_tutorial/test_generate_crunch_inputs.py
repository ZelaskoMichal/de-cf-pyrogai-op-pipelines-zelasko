"""Tests for the generate_crunch_inputs step."""
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import pytest

from template_pipelines.steps.crunch_tutorial.generate_crunch_inputs import GenerateCrunchInputs


@pytest.fixture(scope="function")
def fixture_generate_crunch_inputs_step():
    """Step fixture."""
    with patch("aif.pyrogai.pipelines.components.environment.PlatformProvider"):
        generate_crunch_inputs_step = GenerateCrunchInputs()
        generate_crunch_inputs_step.outputs = {"sales_data": None}
        yield generate_crunch_inputs_step


def test_generate_crunch_inputs_step_run(fixture_generate_crunch_inputs_step):
    """Test run."""
    with does_not_raise():
        fixture_generate_crunch_inputs_step.run()
