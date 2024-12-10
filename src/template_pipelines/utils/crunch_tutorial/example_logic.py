"""Example logic for processing data in CRUNCH job.

You can import any module from src/ folder to use in CRUNCH jobs.
Do not import from outside of src/ folder, it will not be available in CRUNCH job
Add your dependencies to CRUNCH-specific requirements file and specify it in config.json
"""
import csv
from pathlib import Path


def process_data(input_path: Path, output_path: Path) -> None:
    """Example function that takes data from input path, processes it, and writes it to output_path."""
    if input_path.is_file():  # If the input path is a file
        process_file(input_path, output_path)
    elif input_path.is_dir():  # If the input path is a directory
        for file_path in input_path.glob("*"):
            relative_path = file_path.relative_to(input_path)
            if file_path.is_file():
                process_file(file_path, output_path / relative_path)
    else:
        raise ValueError("Invalid input path provided.")


def process_file(input_path: Path, output_path: Path) -> None:
    """Helper function to process a single file."""
    with open(input_path, "r") as input_file:
        reader = csv.DictReader(input_file, delimiter=";")
        Path.mkdir(output_path.parent, parents=True, exist_ok=True)
        with open(output_path, "w") as output_file:
            writer = csv.DictWriter(output_file, delimiter=";", fieldnames=reader.fieldnames)  # type: ignore
            writer.writeheader()
            for row in reader:
                if int(row["Sales"]) > 0:
                    writer.writerow(row)
