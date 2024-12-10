"""Trigger pipeline."""
import time
from typing import Any, Dict, List, Tuple

from aif.pyrogai.steps.step import Step
from aif.rhea.azureml2 import AzureMLPipelineProvider
from aif.rhea.dbr import DbrPipelineProvider
from aif.rhea.run import RunStatus
from aif.rhea.vertex import VertexPipelineProvider


class TriggerPipelinesStep(Step):
    """TriggerPipelinesStep for managing pipeline execution across different platforms."""

    def __init__(self, *args, **kwargs):
        """init."""
        super().__init__(*args, **kwargs)
        self.provider_map = {
            "aml": self.get_aml_provider,
            "dbr": self.get_dbr_provider,
            "vertex": self.get_vertex_provider,
        }

    def get_aml_provider(self) -> AzureMLPipelineProvider:
        """Get Azure ML provider."""
        config_aml = self.config["config_aml"]
        config_aml["service_principal"].update(
            {"sp_id": self.secrets["AML-APP-SP-ID"], "sp_pass": self.secrets["AML-APP-SP-SECRET"]}
        )
        return AzureMLPipelineProvider(config=config_aml)

    def get_dbr_provider(self) -> DbrPipelineProvider:
        """Get Databricks provider."""
        config_dbr = self.config["config_dbr"]
        config_dbr["token"] = self.secrets["dbr_token"]
        return DbrPipelineProvider(config=config_dbr)

    def get_vertex_provider(self) -> VertexPipelineProvider:
        """Get Vertex AI provider."""
        return VertexPipelineProvider(config=self.config["config_vertex"])

    def get_provider(self) -> Any:
        """Get the appropriate provider based on the destination platform."""
        platform = self.platform.lower() if hasattr(self, "platform") else "local"

        if platform == "local":
            dest_platform = self.runtime_parameters.get("dest_platform", "").lower()
            if not dest_platform:
                raise ValueError("Missing dest_platform. Please choose a cloud platform.")
            provider_func = self.provider_map.get(dest_platform)
        else:
            provider_func = self.provider_map.get(platform)

        if not provider_func:
            raise ValueError(f"Unsupported platform: {platform}")

        return provider_func()

    def run_pipelines(
        self,
        provider: Any,
        pipeline: Any,
        config_orchestration: Dict[str, Any],
        config_scopes: Dict[str, Any],
    ) -> Tuple[List[Any], Dict[str, Dict[str, Any]]]:
        """Run pipelines for all scopes."""
        runs = []
        scope_run_map = {}
        for scope in config_orchestration["scopes"]:
            scope_config = config_scopes[scope]
            experiment_name = f"{config_orchestration['pipeline_to_trigger']}_{scope}_exp"
            parameters = {"scope": scope, **scope_config}
            run = provider.submit(
                pipeline=pipeline, experiment_name=experiment_name, parameters=parameters
            )
            runs.append(run)
            scope_run_map[run.run_id] = {"scope": scope, "scope_config": scope_config}
        return runs, scope_run_map

    def wait_for_all_runs(
        self, provider: Any, runs: List[Any], sec_of_waiting_before_next_check: int = 10
    ) -> List[Any]:
        """Wait for all runs to complete."""
        done_runs = []
        satisfactory_statuses = {RunStatus.SUCCEEDED, RunStatus.FAILED}

        while runs:
            time.sleep(sec_of_waiting_before_next_check)
            for run in runs[:]:  # Iterate over a copy of the list
                self.logger.info(f"Checking run {run.run_id}")
                updated_run = provider.get_run(run_id=run.run_id)
                self.logger.info(
                    f"Got run info: run_id={updated_run.run_id}, status={updated_run.status}"
                )

                if updated_run.status in satisfactory_statuses:
                    self.logger.info(
                        f"Run {updated_run.run_id} is finished with status {updated_run.status}"
                    )
                    done_runs.append(updated_run)
                    runs.remove(run)
                    self.log_run_status(runs, done_runs)
                else:
                    self.logger.info(
                        f"Run {updated_run.run_id} isn't finished, waiting another {sec_of_waiting_before_next_check}s"
                    )

        return done_runs

    def log_run_status(self, remaining_runs: List[Any], done_runs: List[Any]) -> None:
        """Log the status of remaining and completed runs."""
        self.logger.info(f"Runs left: {[r.run_id for r in remaining_runs]}")
        self.logger.info(f"Runs done: {[r.run_id for r in done_runs]}")

    def get_summary(
        self, done_runs: List[Any], scope_run_map: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get summary of runs."""
        summary: Dict[str, Any] = {
            "all_runs": len(done_runs),
            "failed": {"qty": 0, "failed_runs": []},
            "passed": {"qty": 0, "passed_runs": []},
        }

        for run in done_runs:
            run_dict = self.prepare_run_dict(run, scope_run_map)
            if run.status == RunStatus.FAILED:
                summary["failed"]["qty"] += 1
                summary["failed"]["failed_runs"].append(run_dict)
            else:
                summary["passed"]["qty"] += 1
                summary["passed"]["passed_runs"].append(run_dict)

        return summary

    def prepare_run_dict(
        self, run: Any, scope_run_map: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare a dictionary representation of a run."""
        return {
            "run_id": run.run_id,
            "status": run.status.name if hasattr(run.status, "name") else str(run.status),
            "pipeline": str(run.pipeline),  # Changed this line to always use str()
            "start_time": str(run.start_time),
            "end_time": str(run.end_time),
            "duration": str(run.duration),
            "scope": scope_run_map[run.run_id],
        }

    def run(self) -> None:
        """Run the step."""
        config_orchestration = self.config["config_orchestration"]
        config_scopes = self.config["config_scopes"]

        self.logger.info(f"Orchestration configuration: {config_orchestration}")
        self.logger.info(f"Scopes configuration: {config_scopes}")

        provider = self.get_provider()
        pipeline_to_trigger = self.get_pipeline_name(config_orchestration["pipeline_to_trigger"])
        pipeline = provider.get_pipeline_by_version(name=pipeline_to_trigger)

        runs, scope_run_map = self.run_pipelines(
            provider, pipeline, config_orchestration, config_scopes
        )
        self.logger.info("Triggered pipelines")
        for r in runs:
            self.logger.info(dict(r))

        done_runs = self.wait_for_all_runs(provider, runs)
        summary = self.get_summary(done_runs, scope_run_map)

        self.logger.info(f"Summary: {summary}")

        self.log_summary_to_mlflow(summary)

    def get_pipeline_name(self, pipeline_name: str) -> str:
        """Get the correct pipeline name based on the platform."""
        if self.platform == "Local":
            if self.runtime_parameters["dest_platform"].lower() in ["aml", "vertex"]:
                return pipeline_name.replace("_", "-")
            return pipeline_name
        else:
            if self.platform in ["AML", "Vertex"]:
                return pipeline_name.replace("_", "-")
            return pipeline_name

    def log_summary_to_mlflow(self, summary: Dict[str, Any]) -> None:
        """Log the summary to MLflow if not using Vertex AI."""
        if (
            self.platform != "Vertex"
            and self.runtime_parameters["dest_platform"].lower() != "vertex"
        ):
            mlflow_client = self.mlflow.MlflowClient()
            pipeline_mlflow_run_id = self.mlflow.get_parent_run(
                self.mlflow.active_run().info.run_id
            ).info.run_id
            mlflow_client.log_dict(pipeline_mlflow_run_id, summary, "summary/data.json")
