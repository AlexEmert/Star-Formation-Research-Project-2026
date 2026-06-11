import pandas as pd
import itertools
import numpy as np
import argparse
from pyhere import here
import os
import matplotlib.pyplot as plt
import pickle
from ratio_function import RatioGenerator
from sklearn.model_selection import train_test_split
from sklearn import set_config
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from skopt.space import Categorical, Real, Integer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from skopt.plots import plot_convergence, plot_objective, plot_evaluations

set_config(transform_output="pandas")

def add_parser_arguments():
    parser = argparse.ArgumentParser(description='Run Bayesian optimization for a model.')
    parser.add_argument('--response', '-r', type=str, required=True, help='The response variable to predict')
    parser.add_argument('--space', '-s', type=str, required=True, help="Name of pickled search space dictionary")
    parser.add_argument('--iters', '-n', type=int, default=10, help='Number of iterations to run the optimization for')
    parser.add_argument('--data', '-d', type=str, default="MIRION_cleaned_everything.csv", help='Name of CSV file containing the data')
    return parser.parse_args()

def main():
    # load data
    args = add_parser_arguments()
    phot = pd.read_csv(here("data/cleaned", args.data))

    # split into X and y -- remove other physical properties
    remove_properties = ['LRATIO', 'T_BOL', 'LM', 'L_BOL', 'MASS', 'DIAM', 'SURF_DENS', 'YB', 'TEMP']

    # remove the one that we want to keep so it isn't dropped
    remove_properties.remove(args.response)

    # only logs the columns that have better log distributions
    if args.response == 'TEMP' or args.response == 'T_BOL':
        y = phot[args.response]
    else:
        y = np.log(phot[args.response])

    X = phot.drop(columns=remove_properties)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=2026)

    flux_cols = ['F1100', 'F870', 'F500', 'F350', 'F250', 'F160', 'F70', 'F24', 'F12', 'F8']

    # reads the space that contains a pipe with preprocessing/model, and a search space created for that model
    with open(here("pipeline/spaces", args.space), "rb") as file:
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

    plt.figure(figsize=(8, 6))
    plot_convergence(opt.optimizer_results_)
    plt.title(f"Convergence Plot: Best {args.response} Model Optimization")
    plt.savefig(here('pipeline/graphing/graphs', f"{args.response}_convergence_plot.png"), dpi=300, bbox_inches='tight')

    #objective plot
    plot_objective(opt.optimizer_results_, size=2)
    plt.title(f"Objective Plot: Best {args.response} Model Optimization")
    plt.savefig(here('pipeline/graphing/graphs', f"{args.response}_objective_plot.png"), dpi=300, bbox_inches='tight')

    # evaluation plot
    plot_evaluations(opt.optimizer_results_, size=2)
    plt.title(f"Evaluation Plot: Best {args.response} Model Optimization")
    plt.savefig(here('pipeline/graphing/graphs', f"{args.response}_evaluation_plot.png"), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()