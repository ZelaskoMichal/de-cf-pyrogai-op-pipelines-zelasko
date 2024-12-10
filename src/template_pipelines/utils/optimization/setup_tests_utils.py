"""Semi-integrated tests base."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Union

from git import Optional

from aif.pyrogai.const import ProviderConfigEnvironments
from aif.pyrogai.pipelines.local.environment import LocalPlatformProvider


def setup_semi_integrated_test(
    config_module: str,
    config_file_name: str,
    pipeline_name: str,
    runtime_parameters: dict,
    step_name: str,
    override_config_path: Optional[Union[Path, str]] = "",
    **kwargs: dict,
) -> None:
    """Set up pre-requisited for individual steps in semi-integrated tests.

    Function to set up the base for running a step with all its features (ioctx, mlflow, etc).
    This creates an instance of the LocalPlatformProvider and set it as the singleton instance
    of the PlatformProvider class, which will be used to build the step when a step class is
    instanced.

    Args:
        config_module (str): config module (e.g. template_pipelines.tests.steps.optimization.config)
        config_file_name (str): name of the base config.json file setting the pipeline
        pipeline_name (str): name to the pipeline being build (needs to match name attribute inside pipeline.yml)
        runtime_parameters (dict): dict of the params to use at runtime
        step_name (str): name of the step
        override_config_path (Optional[Union[Path, str]]): path to config file that overrides config_file_name
        kwargs: expose any other optional arguments to the provider
    """
    # prepare provider
    provider_kwargs = {
        "scope": kwargs.get("scope", ""),
        "environment": ProviderConfigEnvironments.LOCAL,
        "config": config_file_name,
        "override": override_config_path,
        "logger_debug": kwargs.get("logger_debug", False),
        "config_module": config_module,
        "pipeline_name": pipeline_name,
        "step_name": step_name,
        "experiment_name": kwargs.get("experiment_name", None),
        "github_deployment_url": kwargs.get("github_deployment_url", None),
        "github_commit_status_url": kwargs.get("github_commit_status_url", None),
        "provider_name": kwargs.get("provider_name", "Test Local Provider"),
        "run_id": f'test-{datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")}',
        "runtime_params": runtime_parameters,
        "run_timestamp": datetime.now(timezone.utc),
    }

    provider_type = LocalPlatformProvider
    provider_type.set_instance(
        **provider_kwargs,
    )
