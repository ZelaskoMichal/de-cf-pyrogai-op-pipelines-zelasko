"""Bandit evaluation step class."""

import os
from typing import Iterator, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from aif.pyrogai.const import Platform
from aif.pyrogai.steps.step import Step
from template_pipelines.utils.rl_advertising.toolkit import (
    generate_user,
    select_ad,
    simulate_ad_click_probs,
    simulate_ad_inventory,
    simulate_click,
    visualize_line_plot,
)


# Define BanditEvaluation class and inherit properties from pyrogai Step class
class BanditEvaluation(Step):
    """Bandit Evaluation step."""

    def round_nearest_multiple(self, number, multiple):
        """Round number to the nearest multiple."""
        return int(np.floor(number / multiple) * multiple)

    def run_evaluation(
        self,
        model: torch.nn.Module,
        generator: Iterator[Tuple[dict, Sequence[float]]],
        ad_click_probs: dict,
        test_size: int,
    ) -> Tuple[Sequence[float], Sequence[float]]:
        """Run advertising simulation and evaluate the contextual multi-armed bandit agent.

        1. Get a list of available ads at that moment.
        2. Generate a user context.
        3. Display a "high-performing" ad using the bandit agent.
        4. Display a random ad for comparison.
        5. Simulate a user's reaction whether the use clicks on the ad or not.
        6. Calculate average rewards.

        Args:
            model (torch.nn.Module): The action value model.
            generator (Iterator[Tuple[dict, Sequence[float]]]): The generator for generating user context.
            ad_click_probs (dict): Ad click probabilities based on the match between ad target and user context.
            test_size (int): The size of the test data.
        """
        bandit_rewards = []
        random_rewards = []
        bandit_total_reward = 0
        random_total_reward = 0
        for i in range(test_size):
            user, context = next(generator)
            ad_inventory = simulate_ad_inventory()
            ad, _ = select_ad(model, context, ad_inventory)
            random_ad = np.random.choice(ad_inventory)
            bandit_click = simulate_click(ad_click_probs, user, ad)
            random_click = simulate_click(ad_click_probs, user, random_ad)
            bandit_total_reward += bandit_click
            random_total_reward += random_click
            bandit_avg_reward_so_far = bandit_total_reward / (i + 1)
            random_avg_reward_so_far = random_total_reward / (i + 1)
            bandit_rewards.append(bandit_avg_reward_so_far)
            random_rewards.append(random_avg_reward_so_far)

        return (bandit_rewards, random_rewards)

    # Pyrogai executes code defined under run method
    def run(self):
        """Running bandit evaluation step."""
        # Log information using self.logger.info from Pyrogai Step class
        self.logger.info("Running contextual multi-armed bandit evaluation...")

        # Get full path of test_df.parquet
        # from some shared location using Pyrogai Step self.ioctx.get_fn
        # Read in test data created in the Preprocessing step
        input_path = self.ioctx.get_fn("data_preprocessing/test_df.parquet")
        test_df = pd.read_parquet(input_path)

        # Run evaluation
        ad_click_probs = simulate_ad_click_probs()
        generator = generate_user(test_df)
        action_value_model = self.mlflow.pytorch.load_model(self.inputs["model_uri"])
        test_size = self.round_nearest_multiple(len(test_df), 100)
        bandit_rewards, random_rewards = self.run_evaluation(
            action_value_model, generator, ad_click_probs, test_size
        )
        df_reward_comparison = pd.DataFrame(
            {"bandit ad display": bandit_rewards, "random ad display": random_rewards}
        )

        # Store a visualized evalution
        output_dir = self.ioctx.get_output_fn("evaluation")
        plot_path = os.path.join(output_dir, "evaluation.png")
        os.makedirs(output_dir, exist_ok=True)
        fig = visualize_line_plot(df_reward_comparison, "# of impressions", "CTR", plot_path)

        # For Vertex, use ioslot to visualize the evaluation in the Vertex AI pipelines UI
        if self.platform == Platform.VERTEX:
            self.outputs["kfp_md_plot"] = fig

        self.mlflow.log_artifact(plot_path, artifact_path="evaluation_plots")

        self.logger.info("Bandit evaluation is done.")
