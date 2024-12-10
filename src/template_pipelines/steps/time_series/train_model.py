"""Train model step class."""

import itertools

import pandas as pd
import statsmodels.api as sm

from aif.pyrogai.steps.step import Step


class TrainModelStep(Step):
    """Train model step."""

    def train_model(self, co2_series: pd.DataFrame):
        """Train model function.

        The ARIMA model (Autoregressive Integrated Moving Average) is used for time series forecasting.
        It is denoted as ARIMA(p, d, q), where:
          p - Auto-regressive part (incorporates past values)
          d - Integrated part (number of differencing steps)
          q - Moving average part (error terms from past observations)

        Seasonal ARIMA is denoted as ARIMA(p,d,q)(P,D,Q)s, where:
          P, D, Q - Seasonal counterparts of p, d, q
          s - Periodicity of the time series (e.g., 4 for quarterly, 12 for yearly)

        Seasonal ARIMA helps in capturing seasonality, trend, and noise in the data.
        """
        # Define the p, d and q parameters to take any value between 0 and 2
        p = d = q = range(0, 2)

        # Generate all different combinations of p, q and q triplets
        pdq = list(itertools.product(p, d, q))

        # Generate all different combinations of seasonal p, q and q triplets
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

        self.logger.info("MODEL TRAIN START")

        min_aic = float("inf")
        min_model = None

        for param in pdq:
            for param_seasonal in seasonal_pdq:
                mod = sm.tsa.statespace.SARIMAX(
                    co2_series,
                    order=param,
                    seasonal_order=param_seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                results = mod.fit(disp=False)
                if results.aic < min_aic:
                    min_aic = results.aic
                    min_model = results

        self.logger.info("MODEL TRAIN END")
        return min_model

    def run(self):
        """Runs step."""
        fn = self.ioctx.get_fn("co2_data.pkl")
        co2_series = pd.read_pickle(fn)

        min_model = self.train_model(co2_series)

        # Save model to artifacts
        model = self.mlflow.statsmodels.log_model(min_model, "min_model")
        self.outputs["model_uri"] = model.model_uri
