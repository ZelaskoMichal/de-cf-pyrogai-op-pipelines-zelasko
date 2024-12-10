"""Script to combine all artifacts."""

import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def read_and_update_json(file_path: Path, combined_data: dict):
    """Reading and updating json file with matrix."""
    with open(file_path, "r") as file:
        data = json.load(file)
        logging.info(data)

        if data == {}:
            return False

        for pipeline, versions in data.items():
            if pipeline not in combined_data:
                combined_data[pipeline] = {}

            for version, compatibility in versions.items():
                prefixed_version = f"p{version}"  # Add "p" prefix to second-level keys

                if prefixed_version not in combined_data[pipeline]:
                    combined_data[pipeline][prefixed_version] = {}

                for sub_version, compatibility_value in compatibility.items():
                    combined_data[pipeline][prefixed_version][
                        f"t{sub_version}"
                    ] = compatibility_value  # Add "t" prefix to third-level keys


def combine_json_files(directory: Path, output_file: str) -> None:
    """Searching all files with .json and combining them."""
    combined_data: dict = {}

    for item in os.listdir(directory):
        item_path = directory / item
        if item_path.is_dir():
            for filename in os.listdir(item_path):
                file_path = item_path / filename
                if file_path.is_file() and filename.endswith(".json"):
                    read_and_update_json(file_path, combined_data)
        elif item_path.is_file() and item.endswith(".json"):
            read_and_update_json(item_path, combined_data)

    with open(output_file, "w") as file:
        json.dump(combined_data, file, indent=4)

    logging.info("Combined JSON data: %s", combined_data)


if __name__ == "__main__":
    artifacts_path = Path(os.getenv("ARTIFACTS_PATH", "artifacts"))
    combine_json_files(artifacts_path, "compatibility_matrix.json")
