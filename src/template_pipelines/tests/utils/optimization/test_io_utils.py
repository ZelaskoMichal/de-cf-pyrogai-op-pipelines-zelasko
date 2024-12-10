"""Module to test the io_utils."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from template_pipelines.utils.optimization.io_utils import (
    cast_numeric_runtime_parameters,
    copy_data_to_parquet,
    load_tables,
    load_values,
    read_excel_or_csv_with_na,
    save_tables,
    save_values,
)


@pytest.fixture(scope="function")
def input_test_data_paths():
    """Fixture for data inputs."""
    test_data_dir = Path("src/template_pipelines/tests/utils/optimization/test_data")
    yield {
        "csv_inputs": [test_data_dir / "dummy_data_1.csv"],
        "parquet_inputs": [test_data_dir / "dummy_data_2.parquet"],
        "txt_inputs": [test_data_dir / "dummy_data_3.txt"],
        "xls_inputs": [test_data_dir / "dummy_data_4.xls"],
        "xlsx_inputs": [test_data_dir / "dummy_data_5.xlsx"],
        "multi_sheet_xlsx_inputs": [test_data_dir / "multi_sheet_dummy_data.xlsx"],
        "mixed_inputs": [
            test_data_dir / "dummy_data_2.parquet",
            test_data_dir / "dummy_data_1.csv",
        ],
        "json_inputs": [
            test_data_dir / "dummy_values.json",
        ],
    }


@pytest.mark.parametrize(
    "input_path_name, expected_df",
    [
        (
            "parquet_inputs",
            pd.DataFrame(data={"name": ["foobar"], "value": [1.1]}),
        ),
        (
            "csv_inputs",
            pd.DataFrame(data={"name": ["foobar"], "value": [1.1]}),
        ),
        (
            "xlsx_inputs",
            pd.DataFrame(data={"name": ["foobar"], "value": [1.1]}),
        ),
    ],
)
def test_load_tables(input_test_data_paths, input_path_name, expected_df):
    """Tests that various files are loaded correctly to df."""
    # prepare
    file_path = input_test_data_paths[input_path_name][0]
    file_name = file_path.stem

    # execute
    data, location_info = load_tables([file_path])

    # assert
    assert location_info == {file_name: file_path}
    pd.testing.assert_frame_equal(data[file_name], expected_df)


def test_load_values(input_test_data_paths):
    """Tests that values are loaded correctly from json file."""
    # prepare
    expected_values_dict = {"my_value": 1}
    file_path = input_test_data_paths["json_inputs"][0]

    # execute
    data = load_values([file_path])

    # assert
    assert data == expected_values_dict


@pytest.mark.parametrize(
    "file_name, file_format, read_method",
    [
        ("dummy_data", "csv", pd.read_csv),
        ("dummy_data", "parquet", pd.read_parquet),
        ("dummy_data", "xlsx", pd.read_excel),
    ],
)
def test_save_tables(file_name, file_format, read_method):
    """Tests that various files are saved correctly from df."""
    # prepare
    df = pd.DataFrame(data={file_name: ["EMEA", "LAC"]})

    # execute
    with tempfile.TemporaryDirectory() as temp_dir:
        save_tables({file_name: df}, Path(temp_dir), file_format=file_format)
        result_df = read_method(Path(temp_dir) / f"{file_name}.{file_format}")

        # assert
        pd.testing.assert_frame_equal(result_df, df)


@pytest.mark.parametrize(
    "raw_data, expected_data",
    [
        ({"a": np.array([1], dtype=np.int_)[0]}, '{"a": 1}'),
        ({"a": np.array([1.1], dtype=np.float64)[0]}, '{"a": 1.1}'),
        ({"a": np.array([1])}, '{"a": [1]}'),
        ({"a": {1, 2}}, '{"a": "{1, 2}"}'),
        ({"a": tuple([1, 2])}, '{"a": [1, 2]}'),
        ({"a": [1, 2]}, '{"a": [1, 2]}'),
        ({"a": np.array([1], dtype=np.bool_)[0]}, '{"a": "true"}'),
    ],
)
def test_save_values(raw_data, expected_data):
    """Tests that various types of data are saved correctly to json file.

    Custom encoder called NpEncoder is involved during save process.
    """
    # prepare
    filename = "foobar.json"

    # execute
    with tempfile.TemporaryDirectory() as temp_dir:
        save_values(raw_data, Path(temp_dir), filename=filename)
        with open(Path(temp_dir) / filename, "r") as f:
            raw_result_values = f.read()

            # assert
            assert raw_result_values == expected_data


@pytest.mark.parametrize(
    "path_name",
    [
        ("csv_inputs"),
        ("xls_inputs"),
        ("xlsx_inputs"),
        ("multi_sheet_xlsx_inputs"),
    ],
)
def test_read_excel_or_csv_with_na(input_test_data_paths, path_name):
    """Tests to ensure "NA" are not treat as None but normal string."""
    # prepare
    input_path = input_test_data_paths[path_name][0]

    # execute
    df = read_excel_or_csv_with_na(input_path)

    # assert
    assert not df.empty
    assert not df.isnull().values.any()


@pytest.mark.parametrize(
    "input_dict",
    [
        {
            "int_value": "1",
            "float_value_1": "12.0",
            "float_value_2": "10.2",
            "string_value": "hello",
        },
        {
            "int_value": 1,
            "float_value_1": 12.0,
            "float_value_2": 10.2,
            "string_value": "hello",
        },
    ],
)
def test_cast_numeric_runtime_parameters(input_dict):
    """Tests that string values from a dict are properly casted to int or float."""
    # prepare
    expected_output = {
        "int_value": 1,
        "float_value_1": 12,
        "float_value_2": 10.2,
        "string_value": "hello",
    }

    # execute
    result_output = cast_numeric_runtime_parameters(input_dict)

    # assert
    assert result_output == expected_output


@pytest.mark.parametrize(
    "inputs_name, expected_files",
    [
        (
            "csv_inputs",
            ["dummy_data_1.parquet"],
        ),  # csv case
        (
            "parquet_inputs",
            ["dummy_data_2.parquet"],
        ),  # parquet case
        (
            "xls_inputs",
            ["dummy_data_4.parquet"],
        ),  # Excel case 1
        (
            "xlsx_inputs",
            ["dummy_data_5.parquet"],
        ),  # Excel case 2
        (
            "mixed_inputs",
            ["dummy_data_1.parquet", "dummy_data_2.parquet"],
        ),  # mixed type, one files
        (
            "multi_sheet_xlsx_inputs",
            ["dummy_data_sheet_1.parquet", "dummy_data_sheet_2.parquet"],
        ),  # Multi-sheet excel case
    ],
)
def test_copy_data_to_parquet_success(
    inputs_name,
    expected_files,
    input_test_data_paths,
):
    """Test copy_data_to_parquet and asserts that these were properly copied to an ioctx location."""
    # prepare
    inputs = input_test_data_paths[inputs_name]  # use data from fixture

    # execute
    with tempfile.TemporaryDirectory() as tmpdirname:
        copy_data_to_parquet(file_paths=inputs, dest_path=Path(tmpdirname))

        # assert
        # check that the files were actually stored in the tmp directory used by ioctx
        for file_name in expected_files:
            parquet_path = Path(tmpdirname) / file_name
            assert os.path.isfile(parquet_path)


def test_copy_data_to_parquet_fail_duplicated_name(input_test_data_paths):
    """Test copy_data_to_parquet when 2 files with the same base name are passed."""
    with pytest.raises(ValueError):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # execute
            copy_data_to_parquet(
                file_paths=[
                    *input_test_data_paths["parquet_inputs"],
                    *input_test_data_paths["parquet_inputs"],
                ],
                dest_path=Path(tmpdirname),
            )


def test_copy_data_to_parquet_fails_when_unsupported_extension(input_test_data_paths):
    """Test copy_data_to_parquet when a file with unsupported extension is passed."""
    with pytest.raises(ValueError, match="Extension: .txt - is not supported."):
        # execute
        copy_data_to_parquet(input_test_data_paths["txt_inputs"], dest_path="./")
