#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from pmdarima import auto_arima, metrics as metr
from pmdarima.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit

# Define a class for the AutoARIMA model
class AutoARIMAModel:
    def __init__(self, max_p=10, max_q=10, train_real = train_real, test_real=test_real ):
        """
        Initialize the AutoARIMAModel object.

        Parameters:
        - max_p: Maximum number of autoregressive (AR) terms in the ARIMA model.
        - max_q: Maximum number of moving average (MA) terms in the ARIMA model.
        """
        self.max_p = max_p
        self.max_q = max_q
        self.pipeline = None
        self.train_real = train_real
        self.test_real = test_real

    # Method to create the model pipeline
    def create_pipeline(self):
        """
        Create an ARIMA model pipeline.

        This method sets up the pipeline for the AutoARIMA model, including data transformations and hyperparameter tuning.
        """
        self.pipeline = Pipeline([
            ('arima', auto_arima(
                y=self.train_real,
                max_p=self.max_p,
                max_q=self.max_q,
                seasonal=False,
                suppress_warnings=True,
                trace=True))  # AutoARIMA model
        ])

    # Method to perform cross-validation
    def perform_cross_validation(self, n_splits=3, test_size=14):
        """
        Perform cross-validation to evaluate the AutoARIMA model.

        Parameters:
        - train_real: Training time series data.
        - n_splits: Number of cross-validation splits.
        - test_size: Size of the test set in each split.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        rmse_scores = []
        mape_scores = []
        smape_scores = []

        for train_index, test_index in tscv.split(self.train_real):
            train_data, test_data = self.train_real.iloc[train_index], self.train_real.iloc[test_index]
            self.pipeline.fit(train_data)

            forecasts = self.pipeline.predict(len(test_data))

            # Check for NaN or infinity in forecasts
            if np.isnan(forecasts).any() or np.isinf(forecasts).any():
                print("Invalid forecasts found in the current fold. Skipping evaluation for this fold.")
                continue

            # Visualize the predicted values alongside the actual values
            plt.figure()
            plt.plot(test_data.index, test_data.values, label='Actual')
            plt.plot(test_data.index, forecasts, label='Predicted')
            plt.title("Actual vs. Predicted")
            plt.xlabel("Time")
            plt.ylabel("Transformed Number of Crimes")
            plt.legend()
            plt.show()

            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(test_data, forecasts))
            rmse_scores.append(rmse)

            # Calculate MAPE
            mape = mean_absolute_percentage_error(test_data, forecasts)
            mape_scores.append(mape)

            # Calculate SMAPE
            smape = metr.smape(test_data, forecasts)
            smape_scores.append(smape)

        # Print average scores
        print(f"Average RMSE: {np.mean(rmse_scores)}")
        print(f"Average MAPE: {np.mean(mape_scores)}")
        print(f"Average SMAPE: {np.mean(smape_scores)}")

    # Method to make final predictions
    def make_final_predictions(self):
        """
        Make final predictions using the trained AutoARIMA model.

        Parameters:
        - train_real: Training time series data.
        - test_real: Test time series data.

        Returns:
        - final_RMSE: Root Mean Square Error of the final predictions.
        - final_MAPE: Mean Absolute Percentage Error of the final predictions.
        - final_SMAPE: Symmetric Mean Absolute Percentage Error of the final predictions.
        - forecasts: Predicted values.
        """
        self.pipeline.fit(self.train_real)
        forecasts = self.pipeline.predict(len(self.test_real))

        # Calculate the evaluation metrics on the test set
        final_RMSE = np.sqrt(mean_squared_error(self.test_real, forecasts))
        final_MAPE = mean_absolute_percentage_error(self.test_real, forecasts)
        final_SMAPE = metr.smape(self.test_real, forecasts)

        return final_RMSE, final_MAPE, final_SMAPE, forecasts

