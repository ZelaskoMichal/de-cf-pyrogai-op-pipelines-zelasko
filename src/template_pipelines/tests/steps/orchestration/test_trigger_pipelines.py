"""Unittests."""
from unittest.mock import MagicMock, Mock, patch

import pytest

from template_pipelines.steps.orchestration.trigger_pipelines import RunStatus, TriggerPipelinesStep


@pytest.fixture
def mock_step():
    """Mock step."""
    with patch(
        "template_pipelines.steps.orchestration.trigger_pipelines.Step.__init__", return_value=None
    ):
        step = TriggerPipelinesStep()
        step.logger = Mock()
        step.mlflow = Mock()
        step.config = {
            "config_orchestration": {
                "pipeline_to_trigger": "test_pipeline",
                "scopes": ["scope1", "scope2"],
            },
            "config_scopes": {"scope1": {"param1": "value1"}, "scope2": {"param2": "value2"}},
            "config_aml": {"service_principal": {}},
            "config_dbr": {},
            "config_vertex": {"test": "config"},
        }
        step.secrets = {
            "AML-APP-SP-ID": "test-id",
            "AML-APP-SP-SECRET": "test-secret",
            "dbr_token": "test-token",
        }
        step.runtime_parameters = {"dest_platform": "aml"}
        step.platform = "AML"
        return step


@pytest.mark.parametrize(
    "provider_name, expected_class, expected_config",
    [
        (
            "aml",
            "AzureMLPipelineProvider",
            {"service_principal": {"sp_id": "test-id", "sp_pass": "test-secret"}},
        ),
        ("dbr", "DbrPipelineProvider", {"token": "test-token"}),
        ("vertex", "VertexPipelineProvider", {"test": "config"}),
    ],
)
def test_get_provider(mock_step, provider_name, expected_class, expected_config):
    """test_get_provider."""
    mock_step.platform = provider_name

    with patch(
        f"template_pipelines.steps.orchestration.trigger_pipelines.{expected_class}"
    ) as mock_provider:
        mock_step.runtime_parameters["dest_platform"] = provider_name
        provider = mock_step.get_provider()

        mock_provider.assert_called_once_with(config=expected_config)
        assert provider == mock_provider.return_value


def test_get_provider_missing_platform(mock_step):
    """test_get_provider_missing_platform."""
    mock_step.runtime_parameters = {}
    mock_step.platform = "local"
    with pytest.raises(ValueError, match="Missing dest_platform. Please choose a cloud platform."):
        mock_step.get_provider()


def test_run_pipelines(mock_step):
    """test_run_pipelines."""
    mock_provider = Mock()
    mock_pipeline = Mock()

    runs, scope_run_map = mock_step.run_pipelines(
        mock_provider,
        mock_pipeline,
        mock_step.config["config_orchestration"],
        mock_step.config["config_scopes"],
    )

    assert len(runs) == 2
    assert len(scope_run_map) == 1
    mock_provider.submit.assert_called()
    assert mock_provider.submit.call_count == 2


@pytest.mark.parametrize("all_finished", [True, False])
def test_wait_for_all_runs(mock_step, all_finished):
    """test_wait_for_all_runs."""
    mock_provider = Mock()
    mock_runs = [Mock(run_id=f"run_{i}") for i in range(3)]

    class MockRun:
        def __init__(self, run_id, status):
            self.run_id = run_id
            self.status = status

    if all_finished:
        mock_provider.get_run.side_effect = [
            MockRun("run_0", RunStatus.SUCCEEDED),
            MockRun("run_1", RunStatus.FAILED),
            MockRun("run_2", RunStatus.SUCCEEDED),
        ]
    else:
        mock_provider.get_run.side_effect = [
            MockRun("run_0", RunStatus.RUNNING),
            MockRun("run_1", RunStatus.SUCCEEDED),
            MockRun("run_2", RunStatus.RUNNING),
            MockRun("run_0", RunStatus.SUCCEEDED),
            MockRun("run_2", RunStatus.SUCCEEDED),
        ]

    with patch("template_pipelines.steps.orchestration.trigger_pipelines.time.sleep"):
        done_runs = mock_step.wait_for_all_runs(
            mock_provider, mock_runs, sec_of_waiting_before_next_check=0
        )

    assert len(done_runs) == 3
    assert all(run.status in {RunStatus.SUCCEEDED, RunStatus.FAILED} for run in done_runs)


def test_get_summary(mock_step):
    """test_get_summary."""

    class MockRun:
        """MockRun."""

        def __init__(self, run_id, status, start_time, end_time, duration):
            """init."""
            self.run_id = run_id
            self.status = status
            self.start_time = start_time
            self.end_time = end_time
            self.duration = duration
            self.pipeline = Mock()

    mock_runs = [
        MockRun("run_0", RunStatus.SUCCEEDED, "2023-01-01", "2023-01-02", "1 day"),
        MockRun("run_1", RunStatus.FAILED, "2023-01-01", "2023-01-02", "1 day"),
    ]
    scope_run_map = {
        "run_0": {"scope": "scope1", "scope_config": {"param1": "value1"}},
        "run_1": {"scope": "scope2", "scope_config": {"param2": "value2"}},
    }

    summary = mock_step.get_summary(mock_runs, scope_run_map)

    assert summary["all_runs"] == 2
    assert summary["passed"]["qty"] == 1
    assert summary["failed"]["qty"] == 1
    assert len(summary["passed"]["passed_runs"]) == 1
    assert len(summary["failed"]["failed_runs"]) == 1

    # Check the contents of the run dictionaries
    passed_run = summary["passed"]["passed_runs"][0]
    failed_run = summary["failed"]["failed_runs"][0]

    assert passed_run["run_id"] == "run_0"
    assert passed_run["status"] == "SUCCEEDED"
    assert isinstance(passed_run["pipeline"], str)
    assert passed_run["start_time"] == "2023-01-01"
    assert passed_run["end_time"] == "2023-01-02"
    assert passed_run["duration"] == "1 day"
    assert passed_run["scope"] == {"scope": "scope1", "scope_config": {"param1": "value1"}}

    assert failed_run["run_id"] == "run_1"
    assert failed_run["status"] == "FAILED"
    assert isinstance(failed_run["pipeline"], str)
    assert failed_run["start_time"] == "2023-01-01"
    assert failed_run["end_time"] == "2023-01-02"
    assert failed_run["duration"] == "1 day"
    assert failed_run["scope"] == {"scope": "scope2", "scope_config": {"param2": "value2"}}


@patch("template_pipelines.steps.orchestration.trigger_pipelines.TriggerPipelinesStep.get_provider")
@patch(
    "template_pipelines.steps.orchestration.trigger_pipelines.TriggerPipelinesStep.run_pipelines"
)
@patch(
    "template_pipelines.steps.orchestration.trigger_pipelines.TriggerPipelinesStep.wait_for_all_runs"
)
@patch("template_pipelines.steps.orchestration.trigger_pipelines.TriggerPipelinesStep.get_summary")
def test_run(
    mock_get_summary, mock_wait_for_all_runs, mock_run_pipelines, mock_get_provider, mock_step
):
    """test_run."""
    mock_provider = Mock()
    mock_get_provider.return_value = mock_provider
    mock_pipeline = Mock()
    mock_provider.get_pipeline_by_version.return_value = mock_pipeline
    mock_runs = [MagicMock(), MagicMock()]
    mock_run_pipelines.return_value = (mock_runs, {})
    mock_wait_for_all_runs.return_value = mock_runs
    mock_get_summary.return_value = {"summary": "data"}

    mock_step.run()

    mock_get_provider.assert_called_once()
    mock_provider.get_pipeline_by_version.assert_called_once_with(name="test-pipeline")
    mock_run_pipelines.assert_called_once()
    mock_wait_for_all_runs.assert_called_once_with(mock_provider, mock_runs)
    mock_get_summary.assert_called_once()
    mock_step.mlflow.MlflowClient.assert_called_once()
    mock_step.mlflow.MlflowClient().log_dict.assert_called_once_with(
        mock_step.mlflow.get_parent_run().info.run_id, {"summary": "data"}, "summary/data.json"
    )


def test_get_pipeline_name(mock_step):
    """test_get_pipeline_name."""
    assert mock_step.get_pipeline_name("test_pipeline") == "test-pipeline"

    mock_step.platform = "DBR"
    mock_step.runtime_parameters["dest_platform"] = "dbr"
    assert mock_step.get_pipeline_name("test_pipeline") == "test_pipeline"


def test_log_summary_to_mlflow(mock_step):
    """test_log_summary_to_mlflow."""
    summary = {"test": "summary"}
    mock_step.log_summary_to_mlflow(summary)

    mock_step.mlflow.MlflowClient.assert_called_once()
    mock_step.mlflow.MlflowClient().log_dict.assert_called_once_with(
        mock_step.mlflow.get_parent_run().info.run_id, summary, "summary/data.json"
    )

    mock_step.platform = "Vertex"
    mock_step.runtime_parameters["dest_platform"] = "vertex"
    mock_step.mlflow.reset_mock()

    mock_step.log_summary_to_mlflow(summary)
    mock_step.mlflow.MlflowClient.assert_not_called()
