#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
import optuna
from darts.models import Prophet
from darts.metrics import mape
from darts.utils.statistics import check_seasonality
from darts.metrics.metrics import mape, rmse, smape
import warnings

# Create a class for Prophet-based time series forecasting
class ProphetModel:
    def __init__(self, train_real, test_real):
        # Initialize the class with training and test data
        self.train_real_dart, self.test_real_dart = self.prepare_data(train_real, test_real)
    
    def prepare_data(self, train_real, test_real):
        # Prepare and format the time series data
        train_real = train_real.reset_index()
        train_real[train_real.columns[0]] = train_real[train_real.columns[0]].astype("datetime64[ns]")
        # Create a Darts TimeSeries object for training data
        if train_real.columns[1]:
            train_real_dart = TimeSeries.from_dataframe(train_real, train_real.columns[0], [train_real.columns[1]])

        test_real = test_real.reset_index()
        test_real[test_real.columns[0]] = test_real[test_real.columns[0]].astype("datetime64[ns]")
        # Create a Darts TimeSeries object for test data
        if test_real.columns[1]:
            test_real_dart = TimeSeries.from_dataframe(test_real, test_real.columns[0], [test_real.columns[1]])

        return train_real_dart, test_real_dart

    def create_prophet_model(self, hyperparameters, holidays=None):
        # Create a Prophet model with specified hyperparameters and holidays
        model = Prophet(holidays=holidays, **hyperparameters)
        return model

    def objective(self, trial):
        # Define hyperparameters to optimize using Optuna
        params = {
            "changepoint_prior_scale": trial.suggest_loguniform("changepoint_prior_scale", 0.001, 10.0),
            "seasonality_prior_scale": trial.suggest_loguniform("seasonality_prior_scale", 0.01, 100.0),
            "holidays_prior_scale": trial.suggest_loguniform("holidays_prior_scale", 0.01, 100.0),
            "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
        }

        mape_values = []

        # Define holidays here, if applicable
        holidays = None  # Replace with your specific holidays configuration

        # Perform time series cross-validation
        n_splits = 3  # Number of cross-validation splits (adjust as needed)
        split_size = len(self.train_real_dart) // n_splits

        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size

            train_split = self.train_real_dart[start_idx:end_idx]
            val_split = self.train_real_dart[end_idx:end_idx + split_size]

            # Check if either train_split or val_split is empty, and skip this split if empty
            if len(train_split) == 0 or len(val_split) == 0:
                print(f"Trial {i} has Empty dataset")
                continue

            # Create the Prophet model with the current hyperparameters
            model = self.create_prophet_model(params, holidays=holidays)

            # Fit the model to the training data
            model.fit(train_split)

            # Forecast on the validation set
            forecast = model.predict(len(val_split))

            # Calculate MAPE for this split
            mape_value = mape(val_split, forecast)/100
            mape_values.append(mape_value)

        # Calculate the mean MAPE across splits
        mean_mape = sum(mape_values) / (n_splits - len(mape_values))  # Avoid division by zero

        return mean_mape

    def perform_hyperparameter_tuning(self, n_trials=3):
        # Perform hyperparameter tuning using Optuna
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)

        # Print the best hyperparameters
        best_params = study.best_params
        print("Best Hyperparameters:")
        print(best_params)

        return best_params

    def train_and_evaluate_model(self, best_params):
        # Create the final Prophet model with the best hyperparameters
        final_prophet_model = self.create_prophet_model(best_params)

        # Fit the final model to the entire training dataset
        final_prophet_model.fit(self.train_real_dart)

        # Make forecasts with the final model
        forecast = final_prophet_model.predict(14)

        # Calculate evaluation metrics (e.g., RMSE, MAPE, SMAPE)
        rmse_value = rmse(self.test_real_dart, forecast)
        mape_value = mape(self.test_real_dart, forecast)/100
        smape_value = smape(self.test_real_dart, forecast)

        print("The best model metrics are:")
        print(f" RMSE: {rmse_value}")
        print(f" MAPE: {mape_value}")
        print(f" SMAPE: {smape_value}")

        # Create a larger figure for visualization
        plt.figure(figsize=(12, 6))

        # Access the DatetimeIndex directly, don't call it as a function
        time_index = self.test_real_dart.time_index

        # Plot the true test data
        plt.plot(time_index, self.test_real_dart.univariate_values(), label='True Test Data', color='blue', linewidth=2)

        # Plot the predicted values
        plt.plot(time_index, forecast.univariate_values(), label='Predicted Values', color='red', linestyle='dashed', linewidth=2)

        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Time Series Forecasting - Predicted vs. True Test Data')
        plt.legend()
        plt.grid(True)
        plt.show()

