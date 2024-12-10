"""Model training step class."""

import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

from aif.pyrogai.const import Platform
from aif.pyrogai.steps.step import Step  # noqa: E402
from template_pipelines.utils.ml_training.toolkit import (  # noqa: E402
    AutoEncoder,
    build_dataloader,
    load_tables,
)

warnings.filterwarnings("ignore")


# Define ModelTraining class and inherit properties from pyrogai Step class
class ModelTraining(Step):
    """Model Training step."""

    def build_model(self, n_features):
        """Build and compile a model."""
        # Build an anomaly detector model object using toolkit.AutoEncoder
        # The Autoencoder architecture is used to reconstruct normal data
        # Reconstructions are compared with original data in terms of losses
        # The distribution of normal losses can be used
        # to classifiy whether new data inputs have losses within the distribution or not
        # Anomalous data should have a different loss distribution
        # Log relevant metadata using mlflow
        anomaly_detector = AutoEncoder(n_features)
        self.mlflow.log_param("input_shape", n_features)

        # Use MAE as a loss function for dimensionality reduction
        # Create an Adam optimizer for faster convergence
        # Utilize Step self.config to get the initial learning rate defined in config.json
        loss_fn = nn.L1Loss()
        optimizer = torch.optim.Adam(
            anomaly_detector.parameters(), lr=self.config["ml_training"]["learning_rate"]
        )

        # Define LearningRateScheduler to decay learning rate over time
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

        # Log model initial parameters using mlflow
        self.mlflow.log_params(
            {
                "init_learning_rate": self.config["ml_training"]["learning_rate"],
                "opt_loss": "mae",
            }
        )

        return anomaly_detector, loss_fn, optimizer, scheduler

    def train_per_epoch(self, dataloader, anomaly_detector, loss_fn, optimizer):
        """Train the anomaly detector model for one epoch."""
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        anomaly_detector.train()
        train_loss = 0

        # Interate over input batches to calculate loss and gradient values
        # Update the model optimizer and clear gradients after each iteration
        for batch, x in enumerate(dataloader):
            x_reconstructed = anomaly_detector(x)
            loss = loss_fn(x_reconstructed, x)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            if batch % 8 == 0:
                current = (batch + 1) * len(x)
                self.logger.info(f"training loss: {loss.item():>7f} [{current:>5d}/{size:>5d}]")

        train_loss /= num_batches
        self.logger.info(f"average training loss: {train_loss:>8f}")

        return train_loss

    def test_per_epoch(self, dataloader, anomaly_detector, loss_fn):
        """Evaluate the anomaly detector model on the test data for one epoch."""
        num_batches = len(dataloader)
        anomaly_detector.eval()
        test_loss = 0

        # Gradient tracking is not needed during evaluation
        with torch.no_grad():
            for x in dataloader:
                x_reconstructed = anomaly_detector(x)
                test_loss += loss_fn(x_reconstructed, x).item()

        test_loss /= num_batches
        self.logger.info(f"test loss: {test_loss:>8f}\n")

        return test_loss

    def run_model_training(self, anomaly_detector, dataset, loss_fn, optimizer, scheduler):
        """Set parameters in model training to values using pyrogai Step self.config."""
        # Start training the model
        # Log other relevant training metadata
        best_loss = np.inf
        counter = 0
        train_losses = []
        test_losses = []

        # Batch data
        train_unredeemed = build_dataloader(
            dataset["train_unredeemed"].values, self.config["ml_training"]["batch_size"]
        )
        test_unredeemed = build_dataloader(
            dataset["test_unredeemed"].values, self.config["ml_training"]["batch_size"]
        )

        # Train and evaluate the anomaly detector for a number of epochs specified in self.config
        for epoch in range(self.config["ml_training"]["epochs"]):
            self.logger.info(
                f"Epoch: {epoch + 1}, Learning Rate: {optimizer.param_groups[0]['lr']}"
            )
            train_loss = self.train_per_epoch(
                train_unredeemed, anomaly_detector, loss_fn, optimizer
            )
            test_loss = self.test_per_epoch(test_unredeemed, anomaly_detector, loss_fn)

            # Track and decay the initial learning rate over time
            scheduler.step()

            # Stop training if no further improvements are seen
            if test_loss < best_loss:
                best_loss = test_loss
                counter = 0
            else:
                counter += 1

            if counter >= self.config["ml_training"]["stop_learning"]:
                break

            train_losses.append(train_loss)
            test_losses.append(test_loss)

        self.mlflow.log_params(
            {
                "epochs": self.config["ml_training"]["epochs"],
                "batch_size": self.config["ml_training"]["batch_size"],
                "monitor": "val_loss",
                "patience": self.config["ml_training"]["stop_learning"],
            }
        )

        # Store a visualized training history
        output_dir = self.ioctx.get_output_fn("trained_model")
        os.makedirs(output_dir, exist_ok=True)
        fig, ax = plt.subplots()
        plt.plot(train_losses, label="Training Loss")
        plt.plot(test_losses, label="Validation Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.title("Training History")
        plt.savefig(os.path.join(output_dir, "history.png"))

        # For Vertex, use ioslot to visualize the training history in the Vertex AI pipelines UI
        if self.platform == Platform.VERTEX:
            self.outputs["kfp_md_plot"] = fig
        plt.close()

        self.mlflow.log_artifact(
            os.path.join(output_dir, "history.png"), artifact_path="trained_model_plots"
        )

        # Save the trained model in mlflow.artifact
        mlinfo = self.mlflow.pytorch.log_model(anomaly_detector, "anomaly_detector")
        output_dir = self.ioctx.get_output_fn("models")
        model_file = os.path.join(output_dir, "anomaly_model.pth")
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Path to model: {model_file}")
        torch.save(anomaly_detector.state_dict(), model_file)
        self.outputs["anomaly_model"] = model_file
        self.outputs["model_uri"] = mlinfo.model_uri

        return mlinfo

    # Pyrogai executes code defined under run method
    def run(self):
        """Run model training step."""
        # Set seed to replicate the results
        seed = self.config["ml_training"]["random_state"]
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Get full path of file to read a pickle file
        # from some shared location using Pyrogai Step self.ioctx.get_fn
        # Read in ready training data created in the ImputationScaling step
        input_path = self.ioctx.get_fns("imputed_scaled/*.parquet")
        dataset = load_tables(input_path, self.logger)
        n_features = len(dataset["train_unredeemed"].columns)

        # Log information using self.logger.info from Pyrogai Step class
        self.logger.info("Building a model and its training components...")
        anomaly_detector, loss_fn, optimizer, scheduler = self.build_model(n_features)
        self.logger.info("The model has been built.")

        self.logger.info("Running model training...")
        mlinfo = self.run_model_training(anomaly_detector, dataset, loss_fn, optimizer, scheduler)
        self.logger.info(f"The model has been trained and saved to: {mlinfo.model_uri}")
