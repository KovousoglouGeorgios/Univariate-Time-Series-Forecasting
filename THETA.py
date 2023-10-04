#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models.forecasting.theta import FourTheta
from darts.utils.utils import SeasonalityMode, ModelMode, TrendMode
from darts.utils.model_selection import train_test_split
from darts.metrics.metrics import mape, rmse, smape
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit

# Define a class for time series forecasting using the Theta model
class THETA_ts_foreacasting:
    def __init__(self, train_real, test_real):
        """
        Initialize the THETA_ts_foreacasting object.

        Parameters:
        - train_real: Training time series data.
        - test_real: Test time series data.
        """
        self.train_real = train_real
        self.test_real = test_real
        self.best_RMSE = float('inf')
        self.best_MAPE = float('inf')
        self.best_SMAPE = float('inf')
        self.rmse_scores = []
        self.mape_scores = []
        self.smape_scores = []
        self.best_model = None
        self.best_params = None
        self.params_grid = {
            'theta': [2, 4, 6],
            'seasonality_period': [None, 7, 30],
            'season_mode': [SeasonalityMode.MULTIPLICATIVE, SeasonalityMode.ADDITIVE, SeasonalityMode.NONE],
            'model_mode': [ModelMode.MULTIPLICATIVE, ModelMode.ADDITIVE],
            'trend_mode': [TrendMode.LINEAR, TrendMode.EXPONENTIAL],
            'normalization': [True, False]
        }

    def prepare_data(self):
        """
        Prepare the training and test data by converting them to Darts TimeSeries objects.
        """
        self.train_real = self.train_real.reset_index()
        self.train_real[self.train_real.columns[0]] = self.train_real[self.train_real.columns[0]].astype("datetime64[ns]")

        if self.train_real.columns[1]:
            self.train_real_dart = TimeSeries.from_dataframe(self.train_real, self.train_real.columns[0], [self.train_real.columns[1]])

        self.test_real = self.test_real.reset_index()
        self.test_real[self.test_real.columns[0]] = self.test_real[self.test_real.columns[0]].astype("datetime64[ns]")

        if self.test_real.columns[1]:
            self.test_real_dart = TimeSeries.from_dataframe(self.test_real, self.test_real.columns[0], [self.test_real.columns[1]])

    def cross_validate(self):
        """
        Perform cross-validation to find the best hyperparameters for the Theta model.
        """
        tscv = TimeSeriesSplit(n_splits=5)

        for i, (train_index, val_index) in enumerate(tscv.split(self.train_real)):
            train_data, val_data = self.train_real.iloc[train_index, :], self.train_real.iloc[val_index, :]
            train_data = TimeSeries.from_dataframe(train_data, self.train_real.columns[0], [self.train_real.columns[1]])
            val_data = TimeSeries.from_dataframe(val_data, self.train_real.columns[0], [self.train_real.columns[1]])

            theta_model = FourTheta()
            grid_search = theta_model.gridsearch(series=train_data,
                                                 val_series=val_data,
                                                 start=0.1,
                                                 parameters=self.params_grid,
                                                 metric=mape)
            current_best_params = grid_search[1]
            current_best_model = FourTheta(**current_best_params)
            current_best_model.fit(train_data)

            forecast = current_best_model.predict(n=len(val_data))
            RMSE_value = rmse(val_data, forecast)
            mape_value = mape(val_data, forecast)
            smape_value = smape(val_data, forecast)

            if RMSE_value < self.best_RMSE and mape_value < self.best_MAPE:
                self.best_MAPE = mape_value
                self.best_RMSE = RMSE_value
                self.best_SMAPE = smape_value
                self.best_model = current_best_model
                self.best_params = current_best_params

            self.rmse_scores.append(RMSE_value)
            self.mape_scores.append(mape_value)
            self.smape_scores.append(smape_value)

            print(f"Iteration {i + 1}: Best MAPE = {self.best_MAPE:.2f}, Best Parameters = {self.best_params}")

    def evaluate_on_test_data(self):
        """
        Evaluate the Theta model on the test data and calculate RMSE, MAPE, and SMAPE metrics.

        Returns:
        - RMSE_value: Root Mean Square Error on the test data.
        - MAPE_value: Mean Absolute Percentage Error on the test data.
        - SMAPE_value: Symmetric Mean Absolute Percentage Error on the test data.
        """
        # Use the best parameters found during cross-validation
        best_model = FourTheta(**self.best_params)
        best_model.fit(self.train_real_dart)

        forecast = best_model.predict(n=len(self.test_real_dart))
        RMSE_value = rmse(self.test_real_dart, forecast)
        MAPE_value = mape(self.test_real_dart, forecast)/100
        SMAPE_value = smape(self.test_real_dart, forecast)

        return RMSE_value, MAPE_value, SMAPE_value

    def plot_forecast(self):
        """
        Plot the forecasted results against the true test data.
        """
        time_index = np.arange(len(self.train_real_dart), len(self.train_real_dart) + len(self.test_real_dart))
        plt.figure(figsize=(12, 6))
        plt.plot(time_index, self.test_real_dart.univariate_values(), label='True Test Data', color='blue', linewidth=2)
        plt.plot(time_index, self.best_model.predict(n=len(self.test_real_dart)).univariate_values(),
                 label='Predicted Values', color='red', linestyle='dashed', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Time Series Forecasting - Predicted vs. True Test Data')
        plt.legend()
        plt.grid(True)
        plt.show()


