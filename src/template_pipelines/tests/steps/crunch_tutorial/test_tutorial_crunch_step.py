"""Tests for TutorialCrunchStep."""
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from unittest.mock import patch

from aif.pyrogai.pipelines.crunch.crunch_utils_job import CrunchStepConfig
from template_pipelines.steps.crunch_tutorial.tutorial_crunch_step import TutorialCrunchStep


def test_tutorial_crunch_step_run():
    """Test crunch_run."""
    crunch_step_config = CrunchStepConfig(
        input=Path("input"),
        output=Path("output"),
        runtime_parameters={"my_runtime_param": "42"},
        config={"my_config_param": "value"},
        secrets={"foo": "bar"},
    )
    with patch(
        "template_pipelines.steps.crunch_tutorial.tutorial_crunch_step.crunch_step_config",
        crunch_step_config,
    ):
        with patch("template_pipelines.steps.crunch_tutorial.tutorial_crunch_step.process_data"):
            with does_not_raise():
                TutorialCrunchStep.crunch_run()
