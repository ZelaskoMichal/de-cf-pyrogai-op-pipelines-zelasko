"""Unit tests for template_pipelines/steps/aml_sweep/sweep_step.py."""

from datetime import datetime, timezone
from types import MappingProxyType
from unittest.mock import Mock, patch

import lightgbm as lgb
import pandas as pd
import pytest

from aif.pyrogai.const import ProviderConfigEnvironments
from aif.pyrogai.pipelines.local.environment import LocalPlatformProvider
from template_pipelines.steps.aml_sweep.sweep_step import MyAmlSweepStep, train_model


def test_train_model():
    """Test train_model()."""
    x_train = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 3, 4, 5, 6]})
    y_train = pd.DataFrame({"c": [1, 1, 0, 1, 0]})
    x_test = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 3, 4, 5, 6]})
    y_test = pd.DataFrame({"c": [1, 1, 0, 1, 0]})

    res = train_model({}, x_train, x_test, y_train, y_test)
    assert isinstance(res, lgb.Booster)


@pytest.fixture(scope="function")
def step_fixture():
    """Fixture returns SweepStep step for semi-integrated tests.

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
        "step_name": "my_sweep",
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

    step = MyAmlSweepStep()
    yield step


def test__split_x_y(step_fixture):
    """Test step._split_x_y()."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "species": [0, 1, 0, 1, 2], "b": [2, 3, 4, 5, 6]})
    x_df, y_df = step_fixture._split_x_y(df)
    assert x_df.shape == (5, 2)
    assert "species" not in x_df.columns
    assert len(y_df) == 5
    assert y_df.name == "species"


@patch("template_pipelines.steps.aml_sweep.sweep_step.read_any")
def test_sweep_step_run(mock_read, step_fixture):
    """Test run()."""
    step_fixture.ioctx.get_fn = Mock()
    step_fixture.ioctx.get_fn.return_value = "/"
    step_fixture.outputs = {}

    mock_read.return_value = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "species": [0, 1, 0, 1, 2],
            "b": [2, 3, 4, 5, 6],
        }
    )

    # only testing that basic happy path runs
    step_fixture.run()

    assert "cf_trial_model" in step_fixture.outputs
    assert "cf_trial_params" in step_fixture.outputs
