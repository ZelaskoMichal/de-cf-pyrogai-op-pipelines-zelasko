"""Resuable methods and classes."""

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

logging.basicConfig(level=logging.INFO)


class BayesianNN(nn.Module):
    """Define a customized Bayesian NN class as an action-value approximator."""

    def __init__(self, n_input, dropout):
        """Initialize Bayesian NN properties with defined layers, activation and dropout."""
        super().__init__()
        self.fc1 = nn.Linear(n_input, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.use_dropout = dropout > 0

    def forward(self, x):
        """Define a Baysian NN model's forward pass to call the model on inputs and return outputs."""
        x = self.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        output = self.sigmoid(self.fc3(x))

        return output


def build_model(n_input, dropout):
    """Build a Bayesian NN model with specified input size and dropout rate."""
    model = BayesianNN(n_input, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = nn.BCELoss()

    return model, optimizer, loss


def train_per_epoch(X, y, model, loss_fn, optimizer):
    """Train a model per epoch."""
    outputs = model(X).reshape(-1)
    train_loss = loss_fn(outputs, y)
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return train_loss.item()


def update_model(X, y, model, optimizer, loss_fn):
    """Update a model by additional training."""
    X = torch.tensor(X, dtype=torch.float32)
    X = X.reshape((X.shape[0], X.shape[2]))
    y = torch.tensor(y, dtype=torch.float32)

    for _ in range(10):
        train_loss = train_per_epoch(X, y, model, loss_fn, optimizer)
        logging.info(f"Training loss: {train_loss}")


def enable_dropout(model):
    """Enable dropout layers during inference for Bayesian approximation purposes."""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def simulate_ad_inventory():
    """Simulate the availability of ad inventory based on education levels."""
    ad_inv_availability = {
        "elementary": 0.85,
        "middle": 0.65,
        "hs-grad": 0.65,
        "undergraduate": 0.85,
        "graduate": 0.75,
    }
    ad_inventory = []
    for edu_level, prob in ad_inv_availability.items():
        if np.random.uniform() < prob:
            ad_inventory.append(edu_level)

    if not ad_inventory:
        ad_inventory = simulate_ad_inventory()

    return ad_inventory


def simulate_ad_click_probs():
    """Simulate probabilietes of clicks for every combination of ad target and actual user education levels.

    If there's a match between ad target and user education levels, click_prob is a probablity of getting a click.
    If there's a mismatch, the probability of a click decreases by delta.
    """
    click_prob = 0.85
    delta = 0.25
    education_levels = {
        "elementary": 1,
        "middle": 2,
        "hs-grad": 3,
        "undergraduate": 4,
        "graduate": 5,
    }
    ad_click_probs = {
        level_1: {
            level_2: max(
                0, click_prob - delta * abs(education_levels[level_1] - education_levels[level_2])
            )
            for level_2 in education_levels
        }
        for level_1 in education_levels
    }

    return ad_click_probs


def simulate_click(ad_click_probs, user, ad):
    """Display ad and simulate a user's click using the uniform distribution to treat outcomes as equally likely."""
    prob = ad_click_probs[ad][user["education"]]
    click = 1 if np.random.uniform() < prob else 0

    return click


def select_ad(model, context, ad_inventory):
    """Select an ad based on the highest click-thru-rate."""
    selected_ad = None
    selected_x = None
    max_action_val = 0
    model.eval()
    enable_dropout(model)
    with torch.no_grad():
        for ad in ad_inventory:
            ad_x = one_hot_ad(ad)
            x = torch.tensor(context + ad_x, dtype=torch.float32).reshape((1, -1))
            action_val_pred = model(x)
            if action_val_pred >= max_action_val:
                max_action_val = action_val_pred
                selected_ad = ad
                selected_x = x

    return selected_ad, selected_x.numpy()


def one_hot_ad(ad):
    """One hot encode an ad."""
    education_levels = ["elementary", "middle", "hs-grad", "undergraduate", "graduate"]
    ad_input = [0] * len(education_levels)
    if ad in education_levels:
        ad_input[education_levels.index(ad)] = 1

    return ad_input


def generate_user(population):
    """Create a generator for sampling user data."""
    while True:
        user = population.sample(1)
        context = user.iloc[:, :-1].values.tolist()[0]
        yield user.to_dict(orient="records")[0], context


def calculate_regret(user, ad_inventory, ad_click_probs, ad_selected):
    """Estimate regret which is the difference between the simulated truth and predicted probabilities."""
    selected_p = 0
    max_p = 0
    for ad in ad_inventory:
        p = ad_click_probs[ad][user["education"]]
        if ad == ad_selected:
            selected_p = p
        if p > max_p:
            max_p = p
    regret = max_p - selected_p

    return regret


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


def visualize_line_plot(df, xlabel, ylabel, output_path):
    """Create a line plot."""
    fig, ax = plt.subplots()
    df.plot(ax=ax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(output_path)
    plt.close()

    return fig
