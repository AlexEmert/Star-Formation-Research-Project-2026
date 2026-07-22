import pandas as pd
from models.predictive_models.ratio_function import RatioGenerator, LogRatioGenerator
import matplotlib.pyplot as plt
import itertools
import numpy as np
import argparse
from pyhere import here
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn import set_config
from skopt import BayesSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from skopt.space import Categorical, Real, Integer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from skopt.plots import plot_convergence, plot_objective, plot_evaluations
from sklearn.metrics import mean_pinball_loss, make_scorer
from sklearn.model_selection import cross_val_score
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

set_config(transform_output="pandas")

def add_parser_arguments():
    parser = argparse.ArgumentParser(description='Run Bayesian optimization for a model.')
    parser.add_argument('--response', '-r', type=str, required=True, help='The response variable to predict')
    parser.add_argument('--iters', '-n', type=int, default=50, help='Number of iterations to run the optimization for')
    parser.add_argument('--data', '-d', type=str, default= "MIRION_cleaned_everything.csv", help='Name of CSV file containing the data')
    return parser.parse_args()

def main():
    # load data
    args = add_parser_arguments()
    phot = pd.read_csv(here("data/cleaned", args.data))

    lock = JournalFileOpenLock("/bsuscratch/alexanderemert/")
    storage = JournalStorage(JournalFileBackend(f"/bsuscratch/alexanderemert/final_optuna_journal_single_model{args.response}.log", lock_obj=lock))

    # remove tail from tbol
    if args.response == "T_BOL":
        phot = phot[phot['T_BOL']<90]

    # log only the ones with the better distribution logged
    if args.response == "TEMP" or args.response == "T_BOL":
        y = phot[args.response]
    else:
        y=np.log(phot[args.response])
    
    # remove all of the response variables from the features
    X = phot[['F8','e_F8','F12','e_F12','F24','e_F24','F70','e_F70','N8', 'N12', 'N24', "N70", "DIST", 'f_MULTI','f_CEXT']]

    flux_cols = ['F8', 'F12', 'F24', 'F70']

    # drop over a certain relative uncertainty threshold
    for wavelength in ['8', '12', '24', '70']:
        X[f'se_F{wavelength}'] = X[f'e_F{wavelength}'] / np.sqrt(X[f'N{wavelength}'])

    # which quantiles the model will predict
    alphas = [0.025, 0.5, 0.975]

    # in order to save the history between trials
    study_name = "quantile_study_7_21_26_final"

    def objective(trial):
        # threshold for dropping values
        error_threshold = trial.suggest_float("error_threshold", 0.1, 5, step=0.25)

        X_threshold = X.copy()

        for wavelength in ['8', '12', '24', '70']:
            X_threshold.loc[X_threshold[f'se_F{wavelength}'] > error_threshold, f'F{wavelength}'] = np.nan
        X_train, X_test, y_train, y_test = train_test_split(X_threshold, y, test_size = 0.2, random_state=2026)

        # pipeline steps
        scalers = trial.suggest_categorical("scalers", ['standard', 'robust', 'none'])

        if scalers == "standard":
            scaler = StandardScaler()
        elif scalers == 'robust':
            scaler = RobustScaler()
        else:
            scaler = 'passthrough'

        ratios = trial.suggest_categorical("ratios", ['norm_ratio', 'log_ratio'])

        if ratios == 'norm_ratio':
            ratio = RatioGenerator(cols=flux_cols)
        else:
            ratio = LogRatioGenerator(cols=flux_cols)

        imputers = trial.suggest_categorical("imputers", ['mean', 'median', 'none'])
        imputer = None

        if imputers == 'mean':
            imputer == SimpleImputer(strategy='mean')
        elif imputers == 'median': 
            imputer == SimpleImputer(strategy='median')
        else:
            imputer == 'passthrough'

        # xgboost parameters
        xgboost_params = {
            "n_estimators": trial.suggest_int("n_estimators", 100,3000,step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.5, log=True),
            "max_depth": trial.suggest_int("max_depth", 3,12, step=1),
            "min_child_weight": trial.suggest_int("min_child_weight", 1,10, step=1),
            "subsample": trial.suggest_float("subsample", 0.5,1, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1, step=0.1),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1, step=0.1),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 100, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 100, log=True),
            "gamma": trial.suggest_float("gamma", 1e-6, 100, log=True),
            'random_state': 2026,
            'verbosity': 0,
            'n_jobs': 1,
            # now I'm going to include the quantile regression that will predict the rest of the MIRION catalog
            'objective': 'reg:quantileerror'
        }

        score_list = []

        for alpha in alphas:
            q_params = xgboost_params.copy()
            q_params["quantile_alpha"] = alpha
            
            # Train model for this specific quantile
            xgboost_model = XGBRegressor(**q_params)
            model_pipe = make_pipeline(imputer, ratio, scaler, xgboost_model)
            model_pipe.fit(X_train, y_train)
            
            pinball_scorer = make_scorer(
                mean_pinball_loss, 
                alpha=alpha, 
                greater_is_better=False
            )

            loss_scores = cross_val_score(model_pipe, X_train, y_train, scoring=pinball_scorer, cv=6, n_jobs=6)
            score_list.append(np.mean(loss_scores))

        mean_overall_loss = np.mean(score_list)
    
        return mean_overall_loss

    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials=int(args.iters), n_jobs=8)


    mean_study_name = "mean_study_7_21_26"

    # mean model (optimizing a single non-quantile regression model)
    def objective(trial):
        # threshold for dropping values
        error_threshold = trial.suggest_float("error_threshold", 0.1, 5, step=0.25)

        X_threshold_mean = X.copy()

        for wavelength in ['8', '12', '24', '70']:
            X_threshold_mean.loc[X_threshold_mean[f'se_F{wavelength}'] > error_threshold, f'F{wavelength}'] = np.nan
        X_train, X_test, y_train, y_test = train_test_split(X_threshold_mean, y, test_size = 0.2, random_state=2026)

        # pipeline steps
        scalers = trial.suggest_categorical("scalers", ['standard', 'robust', 'none'])

        if scalers == "standard":
            scaler = StandardScaler()
        elif scalers == 'robust':
            scaler = RobustScaler()
        else:
            scaler = 'passthrough'

        ratios = trial.suggest_categorical("ratios", ['norm_ratio', 'log_ratio'])

        if ratios == 'norm_ratio':
            ratio = RatioGenerator(cols=flux_cols)
        else:
            ratio = LogRatioGenerator(cols=flux_cols)

        imputers = trial.suggest_categorical("imputers", ['mean', 'median', 'none'])
        imputer = None

        if imputers == 'mean':
            imputer == SimpleImputer(strategy='mean')
        elif imputers == 'median': 
            imputer == SimpleImputer(strategy='median')
        else:
            imputer == 'passthrough'

        # xgboost parameters
        xgboost_params_single_model = {
            "n_estimators": trial.suggest_int("n_estimators", 100,3000,step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.5, log=True),
            "max_depth": trial.suggest_int("max_depth", 3,12, step=1),
            "min_child_weight": trial.suggest_int("min_child_weight", 1,10, step=1),
            "subsample": trial.suggest_float("subsample", 0.5,1, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1, step=0.1),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1, step=0.1),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 100, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 100, log=True),
            "gamma": trial.suggest_float("gamma", 1e-6, 100, log=True),
            'random_state': 2026,
            'verbosity': 0,
            'n_jobs': 1
        }

        xgboost_model = XGBRegressor(**xgboost_params_single_model)
        model_pipe = make_pipeline(imputer, ratio, scaler, xgboost_model)
        model_pipe.fit(X_train, y_train)

        scores = cross_val_score(model_pipe, X_train, y_train, scoring='neg_root_mean_squared_error', cv=6, n_jobs=6)
        mean_rmse = np.mean(scores)
    
        return mean_rmse

    mean_study = optuna.create_study(direction='maximize', study_name=mean_study_name, storage=storage, load_if_exists=True)
    mean_study.optimize(objective, n_trials=240, n_jobs=8)

    results = {}
    results['best_quantile_error'] = study.best_value
    results['best_quantile_params'] = study.best_params
    results['best_mean_error'] = mean_study.best_value
    results['best_mean_params'] = mean_study.best_params

    with open(here("pipeline/scripts/optuna/final_preds_and_models/results", f"{args.response}_final_predictive_models.pkl"), "wb") as file:
        pickle.dump(results, file)


if __name__ == "__main__":
    main()