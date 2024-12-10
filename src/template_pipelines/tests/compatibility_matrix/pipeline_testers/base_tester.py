"""File with base tester class for inheritance."""

import os
import subprocess
from abc import ABC, abstractmethod
from logging import Logger

import pandas as pd
from compatibility_matrix_controller import CompatibilityMatrixController


class IPipelineTester(ABC):
    """Base tester."""

    def __init__(self, logger: Logger, matrix_controller: CompatibilityMatrixController) -> None:
        """Init."""
        self.logger = logger
        self.matrix_controller = matrix_controller
        self._already_created_files: bool = False

    def prepare_and_test(self) -> None:
        """Main function for preparing env and testing pipeline."""
        self.logger.info(
            f"Start preparation to test {self.matrix_controller.pipeline_name} on pyrogai {self.matrix_controller.pyrogai_version}"  # noqa
        )
        tpt_tags = list(self.matrix_controller.get_tpt_tags_to_test())

        if tpt_tags:
            self.update_pyrogai_version_in_requirements()
            self.install_pyrogai()

            for tpt_tag in tpt_tags:
                self.logger.info(
                    f"Start preparation to test {self.matrix_controller.pipeline_name} on tpt {tpt_tag}"
                )
                try:
                    self.get_pipeline(tpt_tag)
                    self.install_pipeline()
                    self.prepare_env_to_run_pipeline()
                    self.run_pipeline()

                    self.matrix_controller.update_matrix(tpt_tag, True)
                    self.logger.info(
                        f"Finish testing {self.matrix_controller.pipeline_name} on tpt {tpt_tag} with POSITIVE result"
                    )
                except Exception as e:
                    self.matrix_controller.update_matrix(tpt_tag, False)
                    self.logger.warning(
                        f"Finish testing {self.matrix_controller.pipeline_name} on tpt {tpt_tag} with NEGATIVE result"
                    )
                    self.logger.warning(e)

        self.matrix_controller.save_to_json()

    def update_pyrogai_version_in_requirements(self) -> None:
        """Updates the Pyrogai version in the requirements file."""
        self.logger.info("Updating reqs with given pyrogai version")
        try:
            with open("requirements.txt", "r") as file:
                lines = file.readlines()

            updated_lines = [
                self._update_line(line, self.matrix_controller.pyrogai_version) for line in lines
            ]

            with open("requirements.txt", "w") as file:
                file.writelines(updated_lines)
        except IOError as e:
            self.logger.error(f"Error updating requirements file: {e}")

    @staticmethod
    def _update_line(line, new_version):
        """Helper method to update a line with the new version."""
        if "de-cf-pyrogai" in line:
            parts = line.rsplit("@", 1)
            parts[-1] = new_version
            line = "@".join(parts)
        return line

    def install_pyrogai(self) -> None:
        """Installs Pyrogai using pip."""
        self.logger.info("Installing Pyrogai")
        try:
            subprocess.run(["pip", "install", "-e", "."], check=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error installing Pyrogai: {e}")

    def get_pipeline(self, tpt_tag):
        """Installs TPT pipeline."""
        self.logger.info(f"Getting {self.matrix_controller.pipeline_name} on {tpt_tag}")
        try:
            subprocess.check_call(
                [
                    "aif",
                    "pipeline",
                    "from-template",
                    "--template-branch",
                    f"{tpt_tag}",
                    "--pipe-name",
                    self.matrix_controller.pipeline_name,
                    "--config-module",
                    "template_pipelines.config",
                ]
            )
        except subprocess.CalledProcessError as e:
            msg = f"Error occurred during TPT pipeline downloading: {e}"
            self.logger.error(msg)
            raise RuntimeError(msg)

    @abstractmethod
    def prepare_env_to_run_pipeline(self):
        """Abstract function, each inheriting class could write for its own needs.

        For example ml_traning needs data.
        """
        pass

    def install_pipeline(self) -> None:
        """Installation pyrogai."""
        self.logger.info(f"Installing {self.matrix_controller.pipeline_name}")
        try:
            subprocess.check_call(
                [
                    "pip",
                    "install",
                    "-r",
                    f"requirements-{self.matrix_controller.pipeline_name}.txt",
                ]
            )
        except subprocess.CalledProcessError as e:
            msg = f"Error occurred during TPT pipeline installation: {e}"
            self.logger.error(msg)
            raise RuntimeError(msg)

    def run_pipeline(self) -> None:
        """Running pipeline for testing purpose."""
        self.logger.info(f"Running pipeline {self.matrix_controller.pipeline_name}")
        try:
            subprocess.check_call(
                [
                    "aif",
                    "pipeline",
                    "run",
                    "--pipelines",
                    self.matrix_controller.pipeline_name,
                    "--environment",
                    "dev",
                    "--config-module",
                    "template_pipelines.config",
                ]
            )
        except subprocess.CalledProcessError as e:
            msg = f"Error occurred during TPT pipeline run: {e}"
            self.logger.error(msg)
            raise RuntimeError(msg)

    def create_files_in_folder(self, data_to_create_files, folder_name) -> None:
        """Creating files in the specified folder for testing."""
        if not self._already_created_files:
            self.logger.info(f"Creating files in './{folder_name}''.")

            data_path = f"./{folder_name}"
            os.makedirs(data_path, exist_ok=True)

            for file_key, data in data_to_create_files.items():
                df = pd.DataFrame(data)
                file_name = f"{file_key}.parquet"
                df.to_parquet(os.path.join(data_path, file_name))

            self.logger.info(f"Files have been created in ./{folder_name}'.")

        self._already_created_files = True
