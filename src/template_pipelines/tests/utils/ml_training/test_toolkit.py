"""Tests for toolkit."""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

import template_pipelines.utils.ml_training.toolkit as toolkit  # noqa I250


def test_auto_encoder():
    """Test AutoEncoder."""
    x = torch.randn((1, 100))

    autoencoder = toolkit.AutoEncoder(n_dim=100)
    encoder_output = autoencoder.encoder(x)
    decoder_output = autoencoder.decoder(encoder_output)
    output = autoencoder(x)

    assert encoder_output.shape == torch.Size([1, 16])
    assert decoder_output.shape == x.shape
    assert output.shape == x.shape
    assert decoder_output.shape == output.shape
    assert isinstance(output, torch.Tensor)
    assert output.dtype == torch.float32


def test_conver_to_tensor():
    """Test convert_to_tensor with a numpy array."""
    array = np.random.randn(1, 100)
    tensor = toolkit.convert_to_tensor(array)

    assert tensor.dtype == torch.float32
    assert tensor.shape == array.shape


def test_build_dataloader():
    """Test build_dataloader with a numpy array."""
    array = np.random.rand(16, 100)
    batch_size = 4
    dataloader = toolkit.build_dataloader(array, batch_size)
    batch = next(iter(dataloader))

    assert batch.dtype == torch.float32
    assert batch.shape == (batch_size, 100)


@patch("template_pipelines.utils.ml_training.toolkit.os")
def test_save_tables(mock_os):
    """Test save_tables with multiple tables."""
    tables = {"df1": MagicMock(), "df2": MagicMock(), "df3": MagicMock()}
    output_directory = "output_dir"

    toolkit.save_tables(tables, output_directory)

    for table_name, mock_df in tables.items():
        mock_df.to_parquet.assert_called_once_with(
            (Path(output_directory) / (table_name + ".parquet")).as_posix()
        )

    mock_os.makedirs.assert_called_once_with(output_directory, exist_ok=True)


@patch("template_pipelines.utils.ml_training.toolkit.pd")
def test_load_single_table(mock_pd):
    """Test load_tables with a single file."""
    input_list = ["file1"]

    res = toolkit.load_tables(input_list)

    mock_pd.read_parquet.assert_called_once_with("file1")
    assert isinstance(res, dict)
    assert "file1" in res.keys()


@patch("template_pipelines.utils.ml_training.toolkit.pd")
def test_load_multiple_tables(mock_pd):
    """Test load_tables with multiple files."""
    input_list = ["file1.parquet", "file2.parquet"]

    res = toolkit.load_tables(input_list)

    calls = [call("file1.parquet"), call("file2.parquet")]
    mock_pd.read_parquet.assert_has_calls(calls, any_order=True)
    assert isinstance(res, dict)
    assert "file1" in res.keys()
    assert "file2" in res.keys()


@patch("template_pipelines.utils.ml_training.toolkit.pd")
def test_load_no_tables(mock_pd):
    """Test load_tables with an empty list."""
    input_list = []

    res = toolkit.load_tables(input_list)

    mock_pd.read_parquet.assert_not_called()
    assert isinstance(res, dict)
    assert len(res) == 0


def test_prefix_columns():
    """Test prefix_columns."""
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    result = toolkit.prefix_columns(df, "prefix_", ["B"])
    assert list(result.columns) == ["prefix_A", "B"]


def test_label_encode():
    """Test label_encode with null values."""
    df = pd.DataFrame({"A": [1, 2, np.nan, 2], "B": [3, np.nan, np.nan, 4]})
    cat = ["A", "B"]

    df, encoders = toolkit.label_encode(df, cat)

    assert df["A"].iloc[0] == 0
    assert df["A"].iloc[1] == 1
    assert pd.isna(df["A"].iloc[2])
    assert df["A"].iloc[3] == 1

    assert df["B"].iloc[0] == 0
    assert pd.isna(df["B"].iloc[1])
    assert pd.isna(df["B"].iloc[2])
    assert df["B"].iloc[3] == 1

    assert "A" in encoders
    assert "B" in encoders
    assert isinstance(encoders["A"], LabelEncoder)
    assert isinstance(encoders["B"], LabelEncoder)


def test_q1():
    """Test q1."""
    x = pd.Series([1, 2, 3, 4])
    result = toolkit.q1(x)
    assert result == 1.75


def test_q3():
    """Test q3."""
    x = pd.Series([1, 2, 3, 4])
    result = toolkit.q3(x)
    assert result == 3.25


def test_left_join_all():
    """Test left_join_all."""
    df1 = pd.DataFrame({"key": ["A", "B"], "val1": [1, 2]})
    df2 = pd.DataFrame({"key": ["A", "B"], "val2": [3, 4]})
    df3 = pd.DataFrame({"key": ["A", "B"], "val3": [5, 6]})
    dfs = [(df1, "key"), (df2, "key"), (df3, "key")]

    result = toolkit.left_join_all(dfs)

    expected = pd.DataFrame({"key": ["A", "B"], "val1": [1, 2], "val2": [3, 4], "val3": [5, 6]})
    pd.testing.assert_frame_equal(result, expected)
