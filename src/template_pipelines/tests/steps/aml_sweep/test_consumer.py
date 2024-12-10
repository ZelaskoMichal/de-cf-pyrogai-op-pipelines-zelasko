"""Unit tests for template_pipelines/steps/aml_sweep/consumer.py."""

from datetime import datetime, timezone
from types import MappingProxyType
from unittest.mock import patch

import pandas as pd
import pytest

from aif.pyrogai.const import ProviderConfigEnvironments
from aif.pyrogai.pipelines.local.environment import LocalPlatformProvider
from template_pipelines.steps.aml_sweep.consumer import SweepConsumer


@pytest.fixture(scope="function")
def fixture_step():
    """Fixture returns Consumer step for unit tests.

    Step is initialized in similar way like PyrogAI does.
    """
    # prepare provider
    provider_kwargs = {
        "scope": "",
        "environment": ProviderConfigEnvironments.LOCAL,
        "config": "config.json",
        "override": "",
        "logger_debug": False,
        "config_module": "template_pipelines.config",
        "pipeline_name": "aml_sweep",
        "step_name": "sweep_consumer",
        "experiment_name": None,
        "github_deployment_url": None,
        "github_commit_status_url": None,
        "provider_name": "Local Provider",
        "run_id": f'test-{datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")}',
        "runtime_params": MappingProxyType(
            {
                "boosting_type": "gbdt",
                "learning_rate": "0.1",
                "metric": "multi_logloss",
                "num_iterations": "16",
                "max_leaf_nodes": "31",
                "random_seed": "42",
                "verbose": "0",
            }  # Runtime parameters are loaded as strings during config parsing by PyrogAI
        ),
        "run_timestamp": datetime.now(timezone.utc),
    }

    provider_type = LocalPlatformProvider
    provider_type.set_instance(
        **provider_kwargs,
    )

    step = SweepConsumer()
    yield step


@patch("template_pipelines.steps.aml_sweep.consumer.open")
@patch("template_pipelines.steps.aml_sweep.consumer.json.load")
@patch("template_pipelines.steps.aml_sweep.consumer.read_any")
def test_consumer_step_run(mock_read_any, mock_json_load, mock_open, fixture_step):
    """Test run()."""
    mock_json_load.return_value = "{'p1': 1, 'p2': 2}"
    fixture_step.inputs = {
        "cf_trial_params": "no file",
        "cf_trial_model": "src/template_pipelines/tests/steps/aml_sweep/test_data/trial_model",
    }
    mock_read_any.return_value = pd.DataFrame(
        {
            "a": [3, 2, 5, 3, 5],
            "b": [4, 2, 6, 7, 4],
            "species": [2, 1, 0, 0, 1],
        }
    )
    fixture_step.ioctx.get_fn = lambda x: x
    fixture_step.run()
    # only testing that basic happy path runs
