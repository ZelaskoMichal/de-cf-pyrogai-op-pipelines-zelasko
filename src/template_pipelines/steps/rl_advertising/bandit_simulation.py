"""Bandit simulation step class."""

import os
from typing import Iterator, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from aif.pyrogai.const import Platform
from aif.pyrogai.steps.step import Step
from template_pipelines.utils.rl_advertising.toolkit import (
    build_model,
    calculate_regret,
    generate_user,
    select_ad,
    simulate_ad_click_probs,
    simulate_ad_inventory,
    simulate_click,
    update_model,
    visualize_line_plot,
)


# Define BanditSimulation class and inherit properties from pyrogai Step class
class BanditSimulation(Step):
    """Bandit Simulation step."""

    def run_simulation(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        generator: Iterator[Tuple[dict, Sequence[float]]],
        ad_click_probs: dict,
        simulation_size: int,
        model_update_freq: int,
    ) -> Sequence[float]:
        """Run advertising simulation and train the contextual multi-armed bandit agent.

        Steps in simulation:
        1. Get a list of available ads at that moment.
        2. Generate a user context and display a "high-performing" ad.
        3. Simulate a user's reaction whether the use clicks on the ad or not.
        4. Update the action-value model every "model_update_freq".
        5. Calculate regret for performance evaluation.

        Args:
            model (torch.nn.Module): The action-value model.
            optimizer (torch.optim.Optimizer): The optimizer for updating the model.
            loss_fn (torch.nn.Module): The loss function for training the model.
            generator (Iterator[Tuple[dict, Sequence[float]]]): The generator for generating user context.
            ad_click_probs (dict): Ad click probabilities based on the match between ad target and user context.
            simulation_size (int): The number of simulations to run.
            model_update_freq (int): The frequency of updating the model.
        """
        np.random.seed(0)
        X = []
        y = []
        regret_vec = []
        total_regret = 0
        np.random.seed(0)
        for i in range(simulation_size):
            if i % 200 == 0:
                self.logger.info(f"# of impressions: {i}")
            user, context = next(generator)
            ad_inventory = simulate_ad_inventory()
            ad, x = select_ad(model, context, ad_inventory)
            click = simulate_click(ad_click_probs, user, ad)
            regret = calculate_regret(user, ad_inventory, ad_click_probs, ad)
            total_regret += regret
            regret_vec.append(total_regret)
            X.append(x)
            y.append(click)
            if (i + 1) % model_update_freq == 0:
                self.logger.info(f"Updating the model at {i+1}")
                update_model(X, y, model, optimizer, loss_fn)
                X = []
                y = []

        return regret_vec

    def save_model(self, model, model_name, output_dir):
        """Save model in mlflow.artifact."""
        mlinfo = self.mlflow.pytorch.log_model(model, model_name)
        model_path = os.path.join(output_dir, f"{model_name}.pth")

        self.logger.info(f"Path to action_value_model: {model_path}")
        torch.save(model.state_dict(), model_path)

        self.outputs[model_name] = model_path
        self.outputs["model_uri"] = mlinfo.model_uri
        self.logger.info(f"The model has been trained and saved to: {mlinfo.model_uri}")

    # Pyrogai executes code defined under run method
    def run(self):
        """Running bandit simulation step."""
        # Log information using self.logger.info from Pyrogai Step class
        self.logger.info("Running contextual multi-armed bandit simulation...")

        # Get full path of train_df.parquet
        # from some shared location using Pyrogai Step self.ioctx.get_fn
        # Read in training data created in the Preprocessing step
        input_path = self.ioctx.get_fn("data_preprocessing/train_df.parquet")
        train_df = pd.read_parquet(input_path)

        # Set up simulation environment
        ad_click_probs = simulate_ad_click_probs()
        df_bandits = pd.DataFrame()
        config = self.config["rl_advertising"]
        dropout_levels = config["dropout_levels"]
        generator = generate_user(train_df)
        models = {}
        context_n = train_df.shape[1] - 1
        ad_input_n = train_df.education.nunique()

        # Run multiple simulations with different dropout values for the action-value model
        for dropout in dropout_levels:
            self.logger.info(f"Exploring Bayesian approximation with dropout: {dropout}")
            model, optimizer, loss_fn = build_model(context_n + ad_input_n, dropout)
            models[dropout] = model
            regret_vec = self.run_simulation(
                model,
                optimizer,
                loss_fn,
                generator,
                ad_click_probs,
                config["simulation_size"],
                config["model_update_freq"],
            )
            df_bandits[f"dropout: {dropout}"] = regret_vec

        # Store a visualized simulation history
        output_dir = self.ioctx.get_output_fn("simulation")
        plot_path = os.path.join(output_dir, "simulation.png")
        os.makedirs(output_dir, exist_ok=True)
        fig = visualize_line_plot(df_bandits, "# of impressions", "Regret", plot_path)

        # For Vertex, use ioslot to visualize the simulation history in the Vertex AI pipelines UI
        if self.platform == Platform.VERTEX:
            self.outputs["kfp_md_plot"] = fig

        self.mlflow.log_artifact(plot_path, artifact_path="simulation_plots")

        # Select and save the optimal action-value model
        best_dropout = dropout_levels[np.argmin(df_bandits.iloc[-1])]
        selected_model = models[best_dropout]
        self.save_model(selected_model, "action_value_model", output_dir)

        self.logger.info("Contextual multi-armed bandit simulation is done.")
