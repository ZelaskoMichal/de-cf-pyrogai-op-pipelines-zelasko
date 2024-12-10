"""Input output utilities for optimization template pipeline."""
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from aif.pyrogai.ioutils.formats import read_any, write_any

logger = logging.getLogger(__name__)

# adding these kwargs for pd.read_csv limits what strings mean null
READ_CSV_NA_KWARGS = {
    "na_values": [
        "",
        "#N/A",
        "#N/A N/A",
        "#NA",
        "-1.#IND",
        "-1.#QNAN",
        "-NaN",
        "-nan",
        "1.#IND",
        "1.#QNAN",
        "<NA>",
        "N/A",
        "NULL",
        "NaN",
        "n/a",
        "nan",
        "null",
    ],
    "na_filter": False,
}


def load_tables(
    file_paths: List[Path], **kwargs: Any
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Path]]:
    """Load list of file paths into dictionary indexed with file name.

    Args:
        file_paths (List[Path]): iterable of paths to individual files within ioctx to read from,
        as returned by ioctx.get_fn methods.
        **kwargs: arguments passed to pandas reading methods.

    Raises:
        ValueError: unsupported file type read

    Returns:
        Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
            - loaded datasets, with keys as file names
            - mapping of file name to location in ioctx
    """
    data = {}
    location_info = {}
    for file_path in file_paths:
        file_name = file_path.stem

        df = read_any(file_path, **kwargs)

        data[file_name] = df
        location_info[file_name] = file_path

        logger.info(f"Loaded table from {file_path}")
    return data, location_info


def save_tables(
    data: Dict[str, pd.DataFrame], path: Path, file_format: str = "parquet", **kwargs: Any
):
    """Save tables to destination directory.

    Args:
        data (Dict[str, pd.DataFrame]): datasets to save
        path (Path): target location
        file_format str: format to write files as, ie "parquet", "csv, "xls"
        **kwargs: arguments passed to pandas writing methods.

    Raises:
        ValueError: unsupported file type requested
    """
    path.mkdir(parents=True, exist_ok=True)

    for name, df in data.items():
        file_path = path / f"{name}.{file_format}"

        write_any(df, file_path, **kwargs)

        logger.info(f"Saved {name} to {file_path}")


def load_values(file_paths: List[Path]) -> Dict[str, float]:
    """Loads a list of files from json into dictionary.

    Args:
        file_paths (List[Path]): iterable of paths to individual files within ioctx to read from,
        as returned by ioctx.get_fn methods.

    Raises:
        ValueError: unsupported file type read

    Returns:
        Dict[str, float]: - loaded dictionary
    """
    data = {}
    for file_path in file_paths:
        with open(file_path, "r") as f:
            data.update(json.load(f))
        logger.info(f"Loaded values from {file_path}")
    return data


def save_values(data: Dict, path: Path, filename: str = "values.json"):
    """Save values to destination directory as json.

    Args:
        data (Dict): values to save
        path (Path): target location
        filename (str): the filename to save as
    """
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / filename
    logger.debug(data)
    with open(file_path, "w") as f:
        f.write(json.dumps(data, cls=NpEncoder))
    logger.info(f"Saved values to {file_path}")


class NpEncoder(json.JSONEncoder):
    """Json encoder covers numpy cases."""

    def default(self, obj: Any):
        """Method where given object is encoded.

        Args:
            obj(Any): object to encode
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return str(obj)
        if isinstance(obj, np.bool_):
            return super().encode(bool(obj))
        return super(NpEncoder, self).default(obj)


def read_excel_or_csv_with_na(path: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
    """Read excel or csv file with NA handling defaults.

    Args:
        path (Union[str, Path]): file path
        **kwargs: Passed as-is to the underlying native Pandas function (e.g. sheet_name)

    Returns:
        pd.DataFrame: data frame
    """
    supported_extensions = [".csv", ".xls", ".xlsx"]
    file_extension = Path(path).suffix

    if file_extension not in supported_extensions:
        raise ValueError(
            f"You can only read {', '.join(supported_extensions)} files with NA handling defaults."
        )

    return read_any(path, **READ_CSV_NA_KWARGS, **kwargs)


def cast_numeric_runtime_parameters(parameters_dict: dict) -> dict:
    """Cast string params to int ot float if they are numbers."""
    for param_name, value in parameters_dict.items():
        try:
            # Try to convert to float
            value_as_float = float(value)
        except ValueError:
            # if it's not a number, we leave it as a string
            pass
        else:
            # this piece should detect if the value comming is an int or float
            parameters_dict[param_name] = (
                int(value_as_float) if value_as_float.is_integer() else value_as_float
            )

    return parameters_dict


def _file_exists_check(path: Path):
    if os.path.exists(path):
        raise ValueError(f"File already exists in path: {path}.")


def copy_data_to_parquet(file_paths: List[Path], dest_path: Path) -> None:
    """Transforms data files to parquet format and copies them to the desired path.

    Files with extension .csv, .parquet, .xls or .xlsx passed via file_paths are loaded to the destiny path,
    (dest_path). The input files can be read from paths in ioslots and they can be stored to locations in ioctx
    when needed.

    Note: When loading excel files, each sheet will be stored as one parquet file and named after its sheet name.

    Args:
        file_paths (List[Path]): list of filepaths to be copied to dest_path
        dest_path (Path): path to the directory where to store data

    Raises:
        ValueError: if extension is not supported
        ValueError: if file already exists in destiny path
    """
    os.makedirs(dest_path, exist_ok=True)

    for file_path in file_paths:
        logger.info(f"Copying data from {file_path} to location in iocontext: {dest_path}")
        file_name = os.path.split(file_path)[1]  # only get file name
        file_suffix = Path(file_path).suffix

        if file_suffix == ".parquet":
            path_to_write = dest_path / file_name
            _file_exists_check(path_to_write)
            logger.info(f"Copying from {file_path} to {path_to_write}")
            shutil.copyfile(file_path, path_to_write)

        elif file_suffix == ".csv":
            df = read_excel_or_csv_with_na(file_path)
            path_to_write = (dest_path / f"{file_name}").with_suffix(".parquet")
            _file_exists_check(path_to_write)
            logger.info(f"Copying from {file_path} to {path_to_write}")
            pd.DataFrame.to_parquet(df, path_to_write)

        # copying one sheet as one parquet file.
        elif file_suffix in [".xls", ".xlsx"]:
            xls = pd.ExcelFile(file_path)
            for sheet_name in xls.sheet_names:
                df = read_excel_or_csv_with_na(file_path, sheet_name=sheet_name)
                if not df.empty:
                    path_to_write = dest_path / f"{sheet_name}.parquet"
                    _file_exists_check(path_to_write)
                    logger.info(f"Copying from {file_path} to {path_to_write}")
                    pd.DataFrame.to_parquet(df, path_to_write)

        else:
            raise ValueError(f"Extension: {file_suffix} - is not supported.")
