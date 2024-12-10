"""IO helpers."""
import json
from tempfile import NamedTemporaryFile

from pandas import DataFrame


def write_csv_file(df: DataFrame, csv_file_name: str):
    """Write CSV file."""
    with NamedTemporaryFile() as tf:
        tf.name = csv_file_name
        with open(tf.name, "w") as fp:
            df.to_csv(fp, index=False)
            fp.flush()
        return tf.name


def write_json_file(data: dict, file_name: str):
    """Write JSON file."""
    with NamedTemporaryFile() as tf:
        tf.name = file_name
        with open(tf.name, "w") as fp:
            json.dump(data, fp)
            fp.flush()
        return tf.name
