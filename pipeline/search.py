import pandas as pd
from ratio_function import RatioGenerator
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

set_config(transform_output="pandas")


## remove other physical properties?
## split into X and y
## train test split
## simple pipeline that will be used for all models
## take the search space and run bayes search
## return the best params and best cv score

def add_parser_arguments():
    parser = argparse.ArgumentParser(description='Run Bayesian optimization for a model.')
    parser.add_argument('--response', '-r', type=str, required=True, help='The response variable to predict')
    parser.add_argument('--space', '-s', type=str, required=True, help="Name of pickled search space dictionary")
    parser.add_argument('--iters', '-n', type=int, default=50, help='Number of iterations to run the optimization for')
    parser.add_argument('--data', '-d', type=str, default= "MIRION_cleaned_everything.csv", help='Name of CSV file containing the data')
    return parser.parse_args()

def main():
    # load data
    args = add_parser_arguments()
    phot = pd.read_csv(here("data/cleaned", args.data))

    # split into X and y -- remove other physical properties
    remove_properties = ['LRATIO', 'T_BOL', 'LM', 'L_BOL', 'MASS', 'DIAM', 'SURF_DENS', 'YB', 'TEMP']

    # remove the one that we want to keep so it isn't dropped
    remove_properties.remove(args.response)

    y = phot[args.response]
    X = phot.drop(columns=remove_properties)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=2026)

    flux_cols = ['F8', 'F12', 'F24', 'F70', 'F160', 'F250', 'F350', 'F500', 'F870', 'F1100']
    flux_cols = flux_cols [::-1]

    with open(here("spaces", args.space), "rb") as file:
        search_space = pickle.load(file)

    opt = BayesSearchCV(
        estimator=search_space["pipe"],
        search_spaces=search_space["space"],
        n_iter=int(args.iters), 
        cv=10,
        scoring='neg_root_mean_squared_error',
        n_jobs=1,
        random_state=2026
    )

    opt.fit(X_train, y_train)

    log_y = np.log(y)

    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X, log_y, test_size = 0.2, random_state=2026)

    log_opt = BayesSearchCV(
        estimator=search_space["pipe"],
        search_spaces=search_space["space"],
        n_iter=int(args.iters), 
        cv=10,
        scoring='neg_root_mean_squared_error',
        n_jobs=1,
        random_state=2026
    )

    log_opt.fit(X_train_log, y_train_log)

    #save this for later
    space_name = os.path.splitext(os.path.basename(args.space_file))[0]

    with open(here("results", f"{space_name}_{args.response}_results.pkl"), "wb") as file:
        pickle.dump({
            "CVscore": opt.best_score_,
            "best_params": opt.best_params_,
            "logCVscore": log_opt.best_score_,
            "logbest_params": log_opt.best_params_
        }, file)


if __name__ == "__main__":
    main()
