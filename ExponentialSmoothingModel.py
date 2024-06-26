#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pmdarima import metrics as metr
from sklearn import metrics
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Define a class for the Exponential Smoothing model
class ExponentialSmoothingModel:
    def __init__(self, train_data):
        """
        Initialize the ExponentialSmoothingModel object.

        Parameters:
        - train_data: Training time series data.
        """
        self.train_data = train_data

    # Method to train the model with hyperparameter tuning
    def train(self, param_grid, n_splits=3):
        """
        Train the Exponential Smoothing model with hyperparameter tuning and cross-validation.

        Parameters:
        - param_grid: Dictionary containing hyperparameter combinations to search.
        - n_splits: Number of cross-validation splits.

        Returns:
        - best_fit_ES: The best-fitted Exponential Smoothing model.
        - best_combination: The best combination of hyperparameters.
        """
        best_RMSE = float('inf')
        best_MAPE = float('inf')
        best_SMAPE = float('inf')
        best_combination = None

        tscv = TimeSeriesSplit(n_splits=n_splits)

        rmse_scores = []
        mape_scores = []
        smape_scores = []

        for train_index, val_index in tscv.split(self.train_data):
            train_data, val_data = self.train_data.iloc[train_index], self.train_data.iloc[val_index]

            for combination in ParameterGrid(param_grid):
                trend = combination.get('trend')
                seasonal = combination.get('seasonal')
                seasonal_periods = combination.get('seasonal_periods')
                damped = combination.get("damped")

                if seasonal is None:
                    seasonal_periods = None
                elif (seasonal is not None) and (seasonal_periods is None):
                    seasonal_periods = 3

                try:
                    fit_ES = ExponentialSmoothing(train_data,
                                                  trend=trend,
                                                  seasonal=seasonal,
                                                  seasonal_periods=seasonal_periods,
                                                  damped=damped,
                                                  ).fit(optimized=True)
                except ValueError:
                    if (seasonal is None) and (trend is not None):
                        fit_ES = ExponentialSmoothing(train_data,
                                                      trend=trend,
                                                      seasonal_periods=None
                                                     ).fit(optimized=True)
                    else:
                        continue  # Skip this combination if fitting fails

                fcst_pred_ES = fit_ES.forecast(len(val_data))

                RMSE = np.sqrt(metrics.mean_squared_error(val_data, fcst_pred_ES))
                mape = mean_absolute_percentage_error(val_data, fcst_pred_ES)
                smape = metr.smape(val_data, fcst_pred_ES)

                if RMSE < best_RMSE and mape < best_MAPE:
                    best_RMSE = RMSE
                    best_MAPE = mape
                    best_SMAPE = smape
                    best_combination = combination
                    best_fit_ES = fit_ES

            # Print the best combination of hyperparameters for the specific dataset.
            print(f"Best Combination:{best_combination}")

            # Get the best scores of every iteration
            rmse_scores.append(best_RMSE)
            mape_scores.append(best_MAPE)
            smape_scores.append(best_SMAPE)

        # Print average scores
        print(f"Average RMSE from cross-validation: {np.mean(rmse_scores)}")
        print(f"Average MAPE from cross-validation: {np.mean(mape_scores)}")
        print(f"Average SMAPE from cross-validation: {np.mean(smape_scores)}")

        # Return the best_fit_ES and best_combination
        return best_fit_ES, best_combination

    # Method to test the model on new data
    def test(self, best_fit_ES, test_data):
        """
        Test the Exponential Smoothing model on new data.

        Parameters:
        - best_fit_ES: The best-fitted Exponential Smoothing model.
        - test_data: Test time series data.

        Returns:
        - best_RMSE: Root Mean Square Error on the test data.
        - best_MAPE: Mean Absolute Percentage Error on the test data.
        - best_SMAPE: Symmetric Mean Absolute Percentage Error on the test data.
        - predicted_values: Predicted values on the test data.
        """
        # Get the predicted values for the test set
        predicted_values = best_fit_ES.forecast(steps=len(test_data))

        # Calculate the evaluation metrics on the test set
        best_RMSE = np.sqrt(metrics.mean_squared_error(test_data, predicted_values))
        best_MAPE = mean_absolute_percentage_error(test_data, predicted_values)
        best_SMAPE = metr.smape(test_data, predicted_values)

        return best_RMSE, best_MAPE, best_SMAPE, predicted_values

