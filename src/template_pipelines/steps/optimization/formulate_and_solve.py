"""Optimization template pipeline step."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Tuple

import pandas as pd
import xpress as xp

from aif.pyrogai.steps.xpress_step import XpressStep
from dnalib.optimization_models.xpress_utils.logs import IncludeXpressLogs
from dnalib.optimization_models.xpress_utils.solution import (
    convert_to_dict_of_dfs,
    get_slack,
    get_solution,
)
from template_pipelines.steps.optimization.formulation.stock_portfolio_optimization import (
    StockPortfolioOptimization,
)
from template_pipelines.steps.optimization.preprocessing.sdm import StocksPortfolioSDM
from template_pipelines.utils.optimization.io_utils import cast_numeric_runtime_parameters


class FormulateAndSolve(XpressStep):
    """This step uses the input data to formulate and solve an Xpress problem."""

    runtime_parameters: dict

    def pre_run(self) -> None:
        """Cast runtime_parameters from string to numeric."""
        self.runtime_parameters = cast_numeric_runtime_parameters(self.runtime_parameters)

        # store mlflow run ids for unit tests
        self.active_mflow_run_id = self.mlflow_utils.get_active_run_id()
        self.root_mlflow_run_id = self.mlflow_utils.get_root_run_id()

        if self.config.get("L3_integration_test", False):
            # persist tracking uri for aml to query in post_run (it gets cleared inside step)
            self._mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    def run(self):
        """Runs step."""
        self.logger.info("Running formulate_and_solve...")
        self.logger.debug(f"The pipeline is running on {self.platform} platform.")

        self.logger.info("Initializing Xpress...")
        xpress_problem = self.init_xpress_problem()

        self.logger.info("Loading parameters...")
        self.mlflow_utils.log(log_to_root_run=True, params=self.runtime_parameters)
        self.logger.debug(f"Runtime parameters: {self.runtime_parameters}")

        self.logger.info("Building formulation...")
        build_start_time = time.process_time()

        self.logger.info("Loading SDM...")
        (
            sdm_tables_data,
            load_sdm_tables_metrics,
            sdm_values_data,
            load_sdm_values_metrics,
        ) = self.load_sdm()
        self.mlflow_utils.log(
            log_to_root_run=True, metrics={**load_sdm_values_metrics, **load_sdm_tables_metrics}
        )
        self.logger.debug(f"Tables available in the SDM: {sdm_tables_data.keys()}")
        self.logger.debug(f"Values available in the SDM: {sdm_values_data.keys()}")

        self.logger.info("Creating formulation...")
        combined_parameters = {**self.runtime_parameters, **sdm_values_data}
        formulation = StockPortfolioOptimization(
            combined_parameters,
            sdm_tables_data["portfolio_stocks"],
            xpress_problem,
        )
        formulation.build_formulation()
        self.mlflow_utils.log(
            log_to_root_run=True, metrics={"build time": time.process_time() - build_start_time}
        )

        # write the formulation to a text file named portfolio in "LP" format
        # this is nice to do for validating/debugging smaller models,
        # for large models this may not be useful
        self.logger.info("Archiving formulation LPfile...")
        self.log_to_lp_file(xpress_problem)

        # solve with enabled infeasibility pre-check - integer infeasibility would be quickly detected
        # using solve_optimization will also analyze xpress solver statuses which by default dont raise errors
        self.logger.info("Solving formulation...")
        solve_start_time = time.process_time()
        self.logger.info("Setting controls...")
        self.set_solver_controls(xpress_problem)
        self.logger.info("Solving...")
        self.solve_optimization(xpress_problem, infeasibility_check=True)
        self.mlflow_utils.log(
            log_to_root_run=True, metrics={"solve time": time.process_time() - solve_start_time}
        )

        self.logger.info("Retrieving solution...")
        output_dict = get_solution(xpress_problem, formulation.variables)
        self.logger.info(f"Raw solution variables:\n {output_dict}")
        solution_dict_of_dfs = convert_to_dict_of_dfs(
            output_dict
        )  # convert dict of vars to indexed dfs
        self.save_solution(
            solution_dict_of_dfs
        )  # save the solution as separate file for each variable

        self.logger.info("Retrieving slack...")
        slack_dict = get_slack(xpress_problem, formulation.constraints)
        self.logger.info(f"Raw constraint slack: {slack_dict}")
        slack_dict_of_dfs = convert_to_dict_of_dfs(
            slack_dict
        )  # convert dict of constraints into dict of dataframes
        self.merge_and_log_slacks(
            slack_dict_of_dfs
        )  # save the slacks as one file for all constraints

        self.logger.info("Optimization model has been formulated and solved.")

    def post_run(self) -> None:
        """Run optional L3 integration tests."""
        if self.config.get("L3_integration_test", False):
            from template_pipelines.tests.steps.optimization.L3.suite_20stocks import (
                IntegrationTestSuite20Stocks,
            )
            from template_pipelines.tests.steps.optimization.L3.suite_20stocksb import (
                IntegrationTestSuite20StocksB,
            )
            from template_pipelines.tests.steps.optimization.L3.utils import (
                link_mlflow_server,
                repair_dbr_ioslot_dir_path,
            )

            self.logger.info(f"running integration test suite for scope {self.scope}")

            if self.scope == "integrationtest_20stocks":
                test_suite = IntegrationTestSuite20Stocks()
            elif self.scope == "integrationtest_20stocksB":
                test_suite = IntegrationTestSuite20StocksB()
            else:
                self.logger.info(f"no integration test suite found for scope {self.scope}")
                return

            aux_data_path = self.inputs.get("integration_test_auxiliary_data_dir")
            if aux_data_path and self.platform == "DBR":
                aux_data_path = repair_dbr_ioslot_dir_path(aux_data_path, self._ioslots_tempdir)

            with link_mlflow_server(self._mlflow_tracking_uri):
                test_suite.run_tests(
                    step_mlflow_run_id=self.active_mflow_run_id,
                    pipeline_mlflow_run_id=self.root_mlflow_run_id,
                    config=self.config,
                    aux_data_path=aux_data_path,
                    ioctx=self.ioctx,
                )

    def merge_and_log_slacks(self, slack_dict_of_dfs: Dict[str, pd.DataFrame]):
        """Merge all slacks into one dataframe and log as mlflow artifact."""
        # merge indexed constraints dfs
        indexed_constraints_df = pd.merge(
            slack_dict_of_dfs["tie_buy_and_fraction_lb"],
            slack_dict_of_dfs["tie_buy_and_fraction_ub"],
            on="name",
            how="outer",
        )
        indexed_constraints_df = indexed_constraints_df.rename(columns={"name": "stock"})

        indexed_constraints_df = pd.merge(
            indexed_constraints_df,
            slack_dict_of_dfs["min_fraction_per_region"],
            left_on="stock",
            right_on="region",
            how="outer",
        )
        # min_number_per_region is flex constraint and can be turned off
        if "min_number_per_region" in slack_dict_of_dfs:
            indexed_constraints_df = pd.merge(
                indexed_constraints_df,
                slack_dict_of_dfs["min_number_per_region"],
                left_on="region",
                right_on="region",
                how="outer",
            )

        # merge non indexed constraints dfs
        dfs_to_merge = [
            slack_dict_of_dfs[name]
            for name in [
                "invest_all",
                "max_number_risky_sum",
                "max_fraction_risky",
                "max_number_stocks",
            ]
            if name
            in slack_dict_of_dfs  # max_number_risky_sum is flex constraint and can be turned off
        ]
        non_indexed_constraint_df = (
            pd.concat(dfs_to_merge, axis=1).stack().reset_index(level=0, drop=True).reset_index()
        )
        non_indexed_constraint_df.columns = ["constraint_name", "slack"]

        # merge all dataframes into one and log as mlflow artifact
        output_df = pd.merge(
            indexed_constraints_df,
            non_indexed_constraint_df,
            left_on="stock",
            right_on="constraint_name",
            how="outer",
        )
        self.mlflow_utils.handle_logging_artifact(
            file_name="slacks.csv",
            mlflow_artifact_path="solved",
            df=output_df,
        )

    def save_solution(self, output_data: pd.DataFrame):
        """Showcases multiple options to persist the output."""
        # save file to mlflow under current step
        solution_df = pd.merge(output_data["fraction"], output_data["buy"])
        output_mlflow_uri = self.mlflow_utils.handle_logging_artifact(
            file_name="solution.csv", mlflow_artifact_path="solved", df=solution_df
        )
        self.logger.debug(
            f"The output has been saved as a mlflow artifact under {output_mlflow_uri}"
        )

        # save the output as a dataframe per variable in ioctx for following steps
        # create the shared solution directory
        output_dir = self.ioctx.get_output_fn(self.config["solution_tmp_dir"])
        os.makedirs(output_dir, exist_ok=True)

        for variable_name, df in output_data.items():
            df.to_parquet(output_dir / f"{variable_name}.parquet", index=True)
            self.logger.info(f"Saved solved/{variable_name}.parquet to ioctx")

    def init_xpress_problem(self) -> xp.problem:
        """Initialize xpress problem, setup base configuration.

        Returns:
            xp.problem: xpress problem
        """
        # initialize the problem
        self.logger.info("Initializing FICO Xpress...")
        xpress_problem = self.xp.problem(name="portfolio")

        # Adding problem to xpress_problems_dict triggers automatic logging metrics to mlflow.
        # The logging is executed in post run stage in XpressStep. If would be more
        # than 1 problem in dict, each metric name will be prefixed by problem's name (problem.name).
        # Keys in xpress_problems_dict allow easily retrieving problems from dict.
        # More info here:
        # https://developerportal.pg.com/docs/default/component/pyrogai/general-information/explainers/what-is-the-xpress-step/
        # https://developerportal.pg.com/docs/default/component/pyrogai/aif.pyrogai.steps.xpress_step/
        self.xpress_problems_dict["portfolio"] = xpress_problem

        IncludeXpressLogs(self.logger, xpress_problem)

        return xpress_problem

    def set_solver_controls(self, xpress_problem):
        """Set and log solver controls.

        Args:
            xpress_problem (xp.problem): xpress problem instance
        """
        xpress_problem.setControl("miprelstop", 1e-4)

        self.mlflow_utils.log(
            log_to_root_run=True, params=xpress_problem.getControl(["miprelstop"])
        )

    def load_sdm(
        self,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Load all sdm tables and values. Launches base table analysis.

        Returns:
            Tuple[Dict[str, pd.DataFrame], Dict[str, float],  Dict[str, float],  Dict[str, float]]:
                - dict of loaded tables
                - flat dict of tables metrics
                - dict of loaded values
                - dict of values metrics
        """
        sdm = StocksPortfolioSDM()
        sdm.load_stored_sdm(path=self.config["sdm_tmp_dir"], ioctx=self.ioctx)
        tables = {}
        tables_metrics = {}
        values = {}
        values_metrics = {"num_loaded_values": 0}

        # validate loaded sdm tables
        for name, value in sdm.sdm_data.items():
            # check if item is a dataframe, it can be also a float (for parameters stored in file)
            if isinstance(value, pd.DataFrame):
                tables_metrics.update(self.get_table_statistics(name, value))
                tables[name] = value
            else:
                values_metrics["num_loaded_values"] += 1
                values[name] = value

        self.logger.info(f"Loaded table metrics: {tables_metrics}")
        self.logger.info(f"Loaded values metrics: {values_metrics}")

        return tables, tables_metrics, values, values_metrics

    def get_table_statistics(self, table_name: str, table: pd.DataFrame) -> dict:
        """Analayze input table, calculate base statistics. Covers per-table naming convention.

        Args:
            table_name (str): name of the table
            table (pd.DataFrame): table object

        Returns:
            dict: metrics
        """
        metrics = {}

        # rows is one statisic we want for every table
        # if there are others, they can be added here
        metrics["rows"] = len(table)

        metrics = {f"{table_name}_{key}": val for key, val in metrics.items()}
        return metrics

    def log_to_lp_file(self, xpress_problem: xp.problem):
        """Log xpress problem as an LP file artifact to mlflow.

        Args:
            xpress_problem (xp.problem): xpress problem instance
        """
        lpfile_uri = self.mlflow_utils.handle_logging_artifact(
            file_name="portfolio.lp",
            mlflow_artifact_path="formulation",
            file_saving_fn=lambda path: xpress_problem.write(path, "lp"),
        )
        self.logger.info(f"The formulation LP file has been saved to {lpfile_uri}")
