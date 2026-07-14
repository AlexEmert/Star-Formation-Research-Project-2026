import pandas as pd
from ratio_function import RatioGenerator, LogRatioGenerator, pinball_loss_function, check_and_expand_bounds_optuna
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
from sklearn.metrics import root_mean_squared_error, r2_score, make_scorer
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
    parser.add_argument('--threshold', '-t', type=float, default=1.5, help='Level of uncertainty over which fluxes are dropped')
    return parser.parse_args()

def main():
    # load data
    args = add_parser_arguments()
    phot = pd.read_csv(here("data/cleaned", args.data))

    lock = JournalFileOpenLock("/bsuscratch/alexanderemert/")
    storage = JournalStorage(JournalFileBackend(f"/bsuscratch/alexanderemert/optuna_journal_single_model{args.response}.log", lock_obj=lock))

    # split into X and y -- remove other physical properties
    remove_properties = [
        'LRATIO', 
        'T_BOL',  
        'LM', 
        'L_BOL', 
        'MASS', 
        'DIAM', 
        'SURF_DENS', 
        'YB', 
        'TEMP' , 'F160', 'F250', 'F350', 'F500', 'F870', 'F1100', 'e_F160', 'e_F250', 'e_F350', 'e_F500', 'e_F870', 'e_F1100' 
    ]

    # remove tail from tbol
    if args.response == "T_BOL":
        phot = phot[phot['T_BOL']<90]

    # log only the ones with the better distribution logged
    if args.response == "TEMP" or args.response == "T_BOL":
        y = phot[args.response]
    else:
        y=np.log(phot[args.response])
    
    # remove all of the response variables from the features
    X = phot.drop(columns=remove_properties)

    # drop over a certain relative uncertainty threshold
    for wavelength in ['8', '12', '24', '70'
                       #, '160', '250', '350', '500', '870', '1100'
                       ]:
        X.loc[X[f'e_F{wavelength}'] > float(args.threshold), f'F{wavelength}'] = np.nan
        X[f'se_F{wavelength}'] = X[f'e_F{wavelength}'] / np.sqrt(X[f'N{wavelength}'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=2026)

    flux_cols = ['F8', 'F12', 'F24', 'F70'
                 #, 'F160', 'F250', 'F350', 'F500', 'F870', 'F1100'
                 ]

    # which quantiles the model will predict
    alphas = [0.025, 0.5, 0.975]

    pinball_scorer = make_scorer(
    pinball_loss_function, 
    greater_is_better=False, 
    alphas=[0.025, 0.5, 0.975]
    )

    # in order to save the history between trials
    study_name = "my-study"
    storage = f"sqlite:///my-{args.response}-single_model-study.db"

    # contains the parameters input into the model. Updated using a custom function to expand search parameters
    # initial_search_space = {
    #         "n_estimators": (100,3000),
    #         'learning_rate': (1e-4, 0.5),
    #         'max_depth': (3,12),
    #         "min_child_weight": (1,10),
    #         'subsample': (0.5,1),
    #         'colsample_bytree': (0.5, 1),
    #         'colsample_bylevel': (0.5, 1),
    #         'reg_alpha': (1e-6, 100),
    #         'reg_lambda': ( 1e-6, 100),
    #         'gamma': (1e-6, 100)
    #         }

    # possible_bounds = {
    #     "n_estimators": (0,10000),
    #     "learning_rate": (0.001,1),
    #     "max_depth": (0,np.inf),
    #     "min_child_weight": (0,np.inf),
    #     "subsample": (0,1),
    #     "colsample_bytree": (0.01, 1),
    #     "colsample_bylevel": (0.01, 1),
    #     "reg_alpha": (1e-20, np.inf),
    #     "reg_lambda": (1e-20, np.inf),
    #     "gamma":(1e-20,np.inf)
    # }


    def objective(trial):
        # space = trial.study.user_attrs['search_space']

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
            'objective': 'reg:quantileerror',
            'quantile_alpha': alphas
        }

        xgboost_model = XGBRegressor(**xgboost_params)

        model_pipe = make_pipeline(imputer, ratio, scaler, xgboost_model)

        scores = cross_val_score(model_pipe, X_train, y_train, scoring=pinball_scorer, cv=8, n_jobs=8)

        # score is the mean pinball error across cross validation folds
        score = sum(scores) / len(scores)

        return score

    study = optuna.create_study(direction='minimize', study_name=study_name, storage=storage, load_if_exists=True)

    # if 'search_space' not in study.user_attrs:
    #     study.set_user_attr('search_space', initial_search_space)

    max_expansions = 10
    number_expansions = 0

    # while number_expansions <= max_expansions:

    study.optimize(objective, n_trials=int(args.iters), n_jobs=8)

        # expanded, new_space = check_and_expand_bounds_optuna(
        #     study, possible_bounds
        # )

        # study.set_user_attr('search_space', new_space)

        # if expanded == False:
        #     break
        # else:
        #     number_expansions +=1
        
    results = {}
    results['best_error'] = study.best_value
    results['best_params'] = study.best_params
    # results['num_expansions'] = number_expansions


    with open(here("pipeline/results", f"{args.response}_t{args.threshold}-single_model.pkl"), "wb") as file:
        pickle.dump(results, file)


if __name__ == "__main__":
    main()