"""Controller to manage compatibility matrix."""

import json
from logging import Logger
from pathlib import Path
from typing import Dict, Generator, Optional


class CompatibilityMatrixController:
    """Controller to manage compatibility matrix."""

    def __init__(
        self,
        compatibility_matrix_file_path: Path,
        logger: Logger,
        pyrogai_version: Optional[str],
        pipeline_name: str,
        tpt_tags: Generator[str, None, None],
    ) -> None:
        """Initialize the controller with file path, logger, version, pipeline name, and tags."""
        self.logger = logger
        self.tpt_tags = tpt_tags
        self.pyrogai_version = pyrogai_version
        self.pipeline_name = pipeline_name
        self.compatibility_matrix_file_path = compatibility_matrix_file_path
        self.compatibility_matrix: Dict = self.get_matrix()

    def _load_json_file(self) -> dict:
        """Load JSON file and process its content."""
        try:
            with open(self.compatibility_matrix_file_path, "r") as file:
                data = json.load(file)
            return self._process_keys(data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Error loading JSON file: {e}")
            return {}

    def _process_keys(self, data: dict) -> dict:
        """Recursively remove 'p' or 't' from the beginning of keys in the dictionary."""
        if isinstance(data, dict):
            return {
                (key[1:] if key.startswith(("p", "t")) else key): self._process_keys(value)
                for key, value in data.items()
            }
        else:
            return data

    def get_matrix(self) -> Dict:
        """Get the compatibility matrix, updating it with available tags."""
        matrix = self._load_json_file()
        pipeline = matrix.setdefault(self.pipeline_name, {})
        pyrogai = pipeline.setdefault(self.pyrogai_version, {})

        for tpt in self.tpt_tags:
            pyrogai.setdefault(tpt, "")

        self.logger.info(
            f"Updated matrix for {self.pipeline_name} on pyrogai version: {self.pyrogai_version}"
        )
        return matrix

    def update_matrix(self, tag: str, result: bool) -> None:
        """Update the matrix with the given tag and result."""
        try:
            self.compatibility_matrix[self.pipeline_name][self.pyrogai_version][tag] = result
        except KeyError as e:
            self.logger.error(f"Error updating matrix: {e}")
        else:
            self.save_to_json()

    def save_to_json(self) -> None:
        """Save the compatibility matrix to a JSON file."""
        filename = "matrix_result.json"
        try:
            with open(filename, "w", encoding="utf-8") as file:
                json.dump(self.compatibility_matrix, file, ensure_ascii=False, indent=2)
            self.logger.info(f"Results saved into {filename}")
        except Exception as e:
            self.logger.error(f"Error during saving results: {e}")

    def get_tpt_tags_to_test(self) -> Generator[str, None, None]:
        """Yield tags that need to be tested."""
        for tag, value in self.compatibility_matrix[self.pipeline_name][
            self.pyrogai_version
        ].items():
            if value == "":
                yield tag
