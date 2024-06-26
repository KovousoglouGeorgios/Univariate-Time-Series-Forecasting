#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from neuralforecast.losses.pytorch import DistributionLoss
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import time
import warnings

class NBEATSModel:
    def __init__(self, train_real, test_real):
        # Initialize the class with training and test data
        self.train_bneat = self.neuro_form_df(train_real)
        self.test_bneat = self.neuro_form_df(test_real)

    def neuro_form_df(self, df):
        # Format the input DataFrame for NeuralForecast
        df_bneat = df.copy()
        df_bneat = df_bneat.reset_index()
        df_bneat = df_bneat.rename(columns={'DATE OCC': 'ds', 'Occurrence Count': 'y'})
        df_bneat["unique_id"] = 0
        cols = ['ds', 'unique_id', 'y']
        df_bneat = df_bneat[cols]
        return df_bneat

    def objective(self, trial):
        # Define hyperparameters to optimize using Optuna
        input_size = trial.suggest_int('input_size', 90, 360)
        n_blocks_season = trial.suggest_int('n_blocks_season', 1, 3)
        n_blocks_trend = trial.suggest_int('n_blocks_trend', 1, 3)
        n_blocks_identity = trial.suggest_int('n_blocks_ident', 1, 3)
        mlp_units_n = trial.suggest_categorical('mlp_units', [32, 64, 128])
        num_hidden = trial.suggest_int('num_hidden', 1, 3)
        n_harmonics = trial.suggest_int('n_harmonics', 1, 5)
        n_polynomials = trial.suggest_int('n_polynomials', 1, 5)
        scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'robust'])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        n_blocks = [n_blocks_season, n_blocks_trend, n_blocks_identity]
        mlp_units = [[mlp_units_n, mlp_units_n]] * num_hidden

        models = [NBEATS(
            h=14,
            input_size=input_size,
            loss=DistributionLoss(distribution='Poisson', level=[90]),
            max_steps=100,
            stack_types=['seasonality', 'trend', 'identity'],
            mlp_units=mlp_units,
            n_blocks=n_blocks,
            learning_rate=learning_rate,
            n_harmonics=n_harmonics,
            n_polynomials=n_polynomials,
            scaler_type=scaler_type,
        )]
        
        mape_values = []

        n_splits = 3
        split_size = len(self.train_bneat) // n_splits

        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size

            train_split = self.train_bneat[start_idx:end_idx]
            val_split = self.train_bneat[end_idx:end_idx + 14]

            if len(train_split) == 0 or len(val_split) == 0:
                print(f"Trial {i} has Empty dataset")
                continue

            nf = NeuralForecast(models=models, freq='D')
            nf.fit(train_split)
        
            forecast = nf.predict().reset_index()
        
            mape_value = mean_absolute_percentage_error(val_split.y, forecast["NBEATS"])
            mape_values.append(mape_value)

        mean_mape = sum(mape_values) / (n_splits - len(mape_values))
            
        return mape_value

    def optimize_hyperparameters(self, n_trials=2):
        # Perform hyperparameter optimization using Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        best_params = study.best_params
        best_evaluation_metric = study.best_value
        return best_params, best_evaluation_metric

    def train_best_model(self, best_params):
        # Train the best NBEATS model with the optimized hyperparameters
        models = [NBEATS(
            h=14,
            input_size=best_params['input_size'],
            loss=DistributionLoss(distribution='Poisson', level=[90]),
            max_steps=100,
            stack_types=['seasonality', 'trend', 'identity'],
            mlp_units=[[best_params['mlp_units'], best_params['mlp_units']]] * best_params['num_hidden'],
            n_blocks=[best_params['n_blocks_season'], best_params['n_blocks_trend'], best_params['n_blocks_ident']],
            learning_rate=best_params['learning_rate'],
            n_harmonics=best_params['n_harmonics'],
            n_polynomials=best_params['n_polynomials'],
            scaler_type=best_params['scaler_type'],
        )]

        best_model = NeuralForecast(models=models, freq='D')
        best_model.fit(self.train_bneat, val_size=14)
        return best_model

    def evaluate_model(self, model):
        # Evaluate the trained model on the test data
        forecast = model.predict(self.test_bneat)
        rmse_value = np.sqrt(mean_squared_error(self.test_bneat.y, forecast.NBEATS))
        mape_value = mean_absolute_percentage_error(self.test_bneat.y, forecast.NBEATS)

        def smape(actual, forecast):
            return 1 / len(actual) * np.sum(2 * np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast)) * 100)

        smape_value = smape(np.array(self.test_bneat.y), np.array(forecast.NBEATS))

        return rmse_value, mape_value, smape_value

    def plot_forecast(self, model):
        # Plot the forecasted values against the true test data
        plt.figure(figsize=(12, 6))
        time_index = self.test_bneat.ds
        forecast = model.predict(self.test_bneat)
        plt.plot(time_index, self.test_bneat.y, label='True Test Data', color='blue', linewidth=2)
        plt.plot(time_index, forecast.NBEATS, label='Predicted Values', color='red', linestyle='dashed', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Time Series Forecasting - Predicted vs. True Test Data')
        plt.legend()
        plt.grid(True)
        plt.show()

