"""Tests for the example logic step."""
import csv
import tempfile
from pathlib import Path

import pytest

from template_pipelines.utils.crunch_tutorial.example_logic import process_data


def test_process_data_with_file():
    """Test processing a single file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "input_file.csv"
        output_file = Path(temp_dir) / "output_file.csv"

        # Create the input file
        with open(input_file, "w") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(["Sales", "Item"])
            writer.writerow(["10", "Item A"])
            writer.writerow(["-5", "Item B"])

        # Process the data
        process_data(input_file, output_file)

        # Assert output file content
        with open(output_file, "r") as file:
            reader = csv.reader(file, delimiter=";")
            output_data = list(reader)

        expected_output = [["Sales", "Item"], ["10", "Item A"]]
        assert output_data == expected_output


def test_process_data_with_directory():
    """Test processing a directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = Path(temp_dir) / "input_dir"
        output_dir = Path(temp_dir) / "output_dir"

        # Create the input directory
        input_dir.mkdir()
        output_dir.mkdir()
        (input_dir / "nested").mkdir()

        # Create input files in the directory
        input_file_1 = input_dir / "input_file_1.csv"
        input_file_2 = input_dir / "input_file_2.csv"

        with open(input_file_1, "w") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(["Sales", "Item"])
            writer.writerow(["10", "Item A"])
            writer.writerow(["-5", "Item B"])

        with open(input_file_2, "w") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(["Sales", "Item"])
            writer.writerow(["7", "Item C"])
            writer.writerow(["0", "Item D"])

        # Process the data
        process_data(input_dir, output_dir)

        # Assert output file content
        with open(output_dir / "input_file_1.csv", "r") as file:
            reader = csv.reader(file, delimiter=";")
            output_data = list(reader)
            expected_output = [["Sales", "Item"], ["10", "Item A"]]
            assert output_data == expected_output

        with open(output_dir / "input_file_2.csv", "r") as file:
            reader = csv.reader(file, delimiter=";")
            output_data = list(reader)
            expected_output = [["Sales", "Item"], ["7", "Item C"]]
            assert output_data == expected_output


def test_process_data_invalid_path():
    """Test processing an invalid path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "non_existent_file.csv"
        output_path = Path(temp_dir) / "output_file.csv"

        with pytest.raises(ValueError) as excinfo:
            process_data(input_path, output_path)
            assert "Invalid input path provided." in str(excinfo.value)
