import pandas as pd
from ratio_function import RatioGenerator, LogRatioGenerator
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
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from skopt.plots import plot_convergence, plot_objective, plot_evaluations
from sklearn.metrics import root_mean_squared_error, r2_score

set_config(transform_output="pandas")


## remove other physical properties?
## split into X and y
## train test split
## simple pipeline with sbatch and bash script (all_jobs.sh and run_job.sh)
## take the search space and run bayes search for all models
## return the best params and best cv score

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
        'TEMP',
        'F160',
        'F250',
        'F350',
        'F500',
        'F870',
        'F1100'
    ]
    

    ## add SNR as a feature
    bands = ['8', '12', '24', '70']
    for band in bands:
        phot[f'SNR_F{band}'] = phot[f'F{band}'] / phot[f'e_F{band}']


    if args.response == "TEMP" or args.response == "T_BOL":
        y = phot[args.response]
    else:
        y=np.log(phot[args.response])
    
    X = phot.drop(columns=remove_properties)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=2026)

    # other flux cols: 'F1100', 'F870', 'F500', 'F350', 'F250', 'F160'
    flux_cols = ['F70', 'F24', 'F12', 'F8']

    with open(here("pipeline/spaces", "space_list.pkl"), "rb") as file:
        search_space_list = pickle.load(file)

    with open(here("pipeline/spaces", "model_list.pkl"), "rb") as file2:
        model_list = pickle.load(file2)

    model_names = ['cat', 'xgb', 'rf', 'tree', 'svr']
    results = {}

    ## best models for plotting and comparison
    best_model_name = ""
    best_overall_score = float('inf')
    best_overall_model = None
    # skopt_plotting_info = None

    for model_num in range(len(model_list)):
        current_model_name = model_names[model_num]

        ## hard coding to change serach space to n_jobs = -1 if it's SVR because SVR can't multithread
        if model_num == 4 or model_num == 3:
            search_jobs = -1
        else:
            search_jobs = 1

        # switch model to fit that of the pipe
        model_pipe = Pipeline([
            ('impute', SimpleImputer()),
            ('ratio', RatioGenerator(cols=flux_cols)),
            ('scale', RobustScaler()),
            ('model', 'passthrough')
        ])
        model_pipe.set_params(model=model_list[model_num])

        opt = BayesSearchCV(
            estimator=model_pipe,
            search_spaces=search_space_list[model_num],
            n_iter=int(args.iters), 
            cv=10,
            scoring='neg_root_mean_squared_error',
            n_jobs=search_jobs,
            random_state=2026
        )

        opt.fit(X_train, y_train)

        if -opt.best_score_ < best_overall_score:
            best_overall_score = -opt.best_score_
            # skopt_plotting_info = opt.optimizer_results_[-1]
            best_model_name = current_model_name
            best_overall_model = opt.best_estimator_

        results[f'{current_model_name}_CV'] = -opt.best_score_
        results[f'{current_model_name}_params'] = opt.best_params_


    results['best_model_name'] = best_model_name
    results['best_CV'] = best_overall_score
    results['best_model'] = best_overall_model
    y_preds = best_overall_model.predict(X_test)
    test_rmse = root_mean_squared_error(y_test, y_preds)
    test_r2 = r2_score(y_test, y_preds)
    results['test_rmse'] = test_rmse
    results['test_r2'] = test_r2
    results['y_preds'] = y_preds

    # with open(here("pipeline/results", f"{args.response}_mirionfluxes_results.pkl"), "wb") as file:
    #     pickle.dump(results, file)

    # skopt_plot = None

    # if args.response == "TEMP" or args.response == "T_BOL":
    #     skopt_plot = skopt_plotting_info
    #     best_model = best_model_name
    # else:
    #     skopt_plot = skopt_log_plotting_info
    #     best_model = best_log_model_name

    # first_plot_path = here('pipeline/graphing/graphs/skopt_graphs', f"{args.response}_convergence_plot.png")
    # first_plot_path.parent.mkdir(parents=True, exist_ok=True)

    # plt.figure(figsize=(8, 6))
    # plot_convergence(skopt_plot)
    # plt.title(f"{args.response} Convergence Plot: {best_model} model")
    # plt.savefig(str(here('pipeline/graphing/graphs/skopt_graphs', f"{args.response}_convergence_plot.png")), dpi=300, bbox_inches='tight')

    # #objective plot
    # plot_objective(skopt_plot, size=2)
    # plt.title(f"{args.response} Objective Plot: {best_model} model")
    # plt.savefig(str(here('pipeline/graphing/graphs/skopt_graphs', f"{args.response}_objective_plot.png")), dpi=300, bbox_inches='tight')

    # # evaluation plot
    # plot_evaluations(skopt_plot, size=2)
    # plt.title(f"{args.response} Evaluation Plot: {best_model} model")
    # plt.savefig(str(here('pipeline/graphing/graphs/skopt_graphs', f"{args.response}_evaluation_plot.png")), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
