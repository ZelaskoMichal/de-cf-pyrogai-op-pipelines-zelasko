"""Resuable methods and classes."""

import os
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder  # noqa: E402
from torch import nn
from torch.utils.data import DataLoader


class AutoEncoder(nn.Module):
    """Define a customized AutoEncoder model."""

    def __init__(self, n_dim):
        """Initialize AutoEncoder properties with defined layers and activation."""
        super().__init__()
        self.n_dim = n_dim

        # Encoder model
        self.encoder = nn.Sequential(
            nn.Linear(self.n_dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU()
        )

        # Decoder model
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, self.n_dim)
        )

    def forward(self, x):
        """Define a model's forward pass to call the model on inputs and return outputs."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def convert_to_tensor(array):
    """Get a pytorch tensor from a numpy array."""
    return torch.tensor(array, dtype=torch.float32)


def build_dataloader(array, batch_size):
    """Define a pytorch dataloader and create batches of numpy inputs."""
    tensor = convert_to_tensor(array)
    dataloader = DataLoader(tensor, batch_size=batch_size)
    return dataloader


def save_tables(tables: dict, output_dir: str, logger=None):
    """Receives a dictionary of DataFrames and saves as individual parquet files.

    Args:
        tables (dict): Dictionary of DataFrames
        output_dir (str): Destination folder
        logger (optional): Step logger for output
    """
    os.makedirs(output_dir, exist_ok=True)
    for name, df in tables.items():
        outfile = (Path(output_dir) / (name + ".parquet")).as_posix()
        if logger:
            logger.info(f"Saving {outfile}")
        df.columns = df.columns.astype(str)
        df.to_parquet(outfile)


def label_encode(df, cat):
    """Label encode values except the missing ones."""
    encoders = {}
    for col in cat:
        encoder = LabelEncoder()
        df.loc[~df[col].isnull(), col] = encoder.fit_transform(df.loc[~df[col].isnull(), col])
        encoders[col] = encoder
    return df, encoders


def load_tables(input_list: str, logger=None):
    """Loads a list of parquet files and returns a dictionary of DataFrames.

    Args:
        input_list (str): A list of file names to load, for instance as
            obtained with ioctx.get_fns
        keys (list, optional): List of tables to be expected. Assumes that file with the names
            listed here exist in input dir with the parquet extension.
            Defaults to ["campaigns", "coupon_item_mapping", "coupon_redemption",
                         "customer_demographics", "customer_transactions", "items"].
        logger (optional): Step logger for output

    Returns:
        Dictionary of tables
    """
    tables = {}
    for file in input_list:
        basename = Path(file).stem
        if logger:
            logger.info(f"Loading {file}")
        tables[basename] = pd.read_parquet(file)
    return tables


def prefix_columns(df, prefix, exclude_columns):
    """Add a prefix columns with exclusion."""
    return df.rename(
        columns={col: prefix + col for col in df.columns if col not in exclude_columns}
    )


def q1(x):
    """Estimate 25% quantile."""
    return x.quantile(0.25)


def q3(x):
    """Estimate 75% quantile."""
    return x.quantile(0.75)


def left_join_all(dfs):
    """Left join a list of dataframes."""
    main_table = dfs[0][0]
    for table, key in dfs[1:]:
        main_table = main_table.merge(table, on=key, how="left")
    return main_table
