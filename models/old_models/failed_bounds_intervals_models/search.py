import pandas as pd
from models.old_models.failed_bounds_intervals_models.ratio_function import RatioGenerator, LogRatioGenerator, check_and_expand_space, pinball_loss_function
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
from sklearn.metrics import root_mean_squared_error, r2_score, make_scorer

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

    
    model_pipe = Pipeline([
            ('impute', SimpleImputer()),
            ('ratio', RatioGenerator(cols=flux_cols)),
            ('scale', RobustScaler()),
            ('model', XGBRegressor(random_state=2026, verbosity=0, n_jobs=1))
        ])
    
    search_space = {
        'impute': Categorical([SimpleImputer(strategy='mean'), SimpleImputer(strategy='median'), KNNImputer(), 'passthrough']),
        'ratio': Categorical(['passthrough', RatioGenerator(cols=flux_cols),LogRatioGenerator(cols=flux_cols)]),
        'scale': Categorical([StandardScaler(), RobustScaler(), 'passthrough']),
        "model__n_estimators": Integer(100, 3000),
        "model__learning_rate": Real(1e-4, 0.5, prior="log-uniform"),
        "model__max_depth": Integer(3, 12),
        "model__min_child_weight": Integer(1, 20),
        "model__subsample": Real(0.5, 1.0, prior="uniform"),
        "model__colsample_bytree": Real(0.5, 1.0, prior="uniform"),
        "model__colsample_bylevel": Real(0.5, 1.0, prior="uniform"),
        "model__reg_alpha": Real(1e-9, 100.0, prior="log-uniform"),
        "model__reg_lambda": Real(1e-9, 100.0, prior="log-uniform"),
        "model__gamma": Real(1e-9, 10.0, prior="log-uniform")
    }

    possible_bounds = {
        "model__n_estimators": (0,10000),
        "model__learning_rate": (0.001,1),
        "model__gamma":(1e-20,np.inf),
        "model__max_depth": (0,np.inf),
        "model__min_child_weight": (0,np.inf),
        "model__subsample": (0,1),
        "model__colsample_bytree": (0.01, 1),
        "model__colsample_bylevel": (0.01, 1),
        "model__reg_alpha": (1e-20, np.inf),
        "model__reg_lambda": (1e-20, np.inf)
    }

    results = {}
    number_expansions=0
    iterations = int(args.iters)

    pinball_scorer = make_scorer(
    pinball_loss_function, 
    greater_is_better=False, 
    alphas=[0.025, 0.5, 0.975]
    )

    while True:

        opt = BayesSearchCV(
            estimator=model_pipe,
            search_spaces=search_space,
            n_iter=iterations, 
            cv=8,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            n_points=4,
            random_state=2026
        )

        opt.fit(X_train, y_train)

        expanded, search_space = check_and_expand_space(
            best_params=opt.best_params_,
            current_space=search_space,
            definitive_bounds=possible_bounds,
            tolerance=0.05, 
            expansion=0.5
        )

        if expanded == True:
            number_expansions += 1
            iterations += 100

        if number_expansions == 15:
            break

        if not expanded:
            break

    
    y_preds = opt.best_estimator_.predict(X_test)
    test_rmse = root_mean_squared_error(y_test, y_preds)
    test_r2 = r2_score(y_test, y_preds)

    results['primary_CV_rmse'] = -opt.best_score_
    results['primary_params'] = opt.best_params_
    results['bounds_expanded_iters'] = number_expansions
    results['test_rmse'] = test_rmse
    results['test_r2'] = test_r2
    results['y_preds'] = y_preds

    residuals = y_train - opt.best_estimator_.predict(X_train)
    # now fit quantile model on the residuals to find the amount to add/subtract for confidence intervals
    alphas = [0.025, 0.975]
    model_pipe = Pipeline([
            ('impute', SimpleImputer()),
            ('ratio', RatioGenerator(cols=flux_cols)),
            ('scale', RobustScaler()),
            ('model', XGBRegressor(random_state=2026, verbosity=0, n_jobs=1, objective="reg:quantileerror", quantile_alpha=alphas))
        ])

    while True:

        quant_opt = BayesSearchCV(
            estimator=model_pipe,
            search_spaces=search_space,
            n_iter=iterations, 
            cv=8,
            scoring=pinball_scorer,
            n_jobs=-1,
            n_points=4,
            random_state=2026
        )

        quant_opt.fit(X_train, y_train)

        expanded, search_space = check_and_expand_space(
            best_params=quant_opt.best_params_,
            current_space=search_space,
            definitive_bounds=possible_bounds,
            tolerance=0.05, 
            expansion=0.5
        )

        if expanded == True:
            number_expansions += 1
            iterations += 100

        if number_expansions == 15:
            break

        if not expanded:
            break


    boundary_predictions = quant_opt.best_estimator_.predict(y_train)

    results['0.025_y_preds'] = y_preds + boundary_predictions[:,0]
    results['0.0975_y_preds'] = y_preds + boundary_predictions[:,1]
    results['bounds'] = boundary_predictions
    results['quantile_model_params'] = quant_opt.best_params_


    with open(here("pipeline/results", f"{args.response}_predictive_results_t{args.threshold}.pkl"), "wb") as file:
        pickle.dump(results, file)


    # if int(args.threshold) == 1.5:
    #     skopt_plot = opt.optimizer_results_[-1]

    #     first_plot_path = here('pipeline/graphing/graphs/skopt_graphs', f"{args.response}_convergence_plot.png")
    #     first_plot_path.parent.mkdir(parents=True, exist_ok=True)

    #     plt.figure(figsize=(8, 6))
    #     plot_convergence(skopt_plot)
    #     plt.title(f"{args.response} Convergence Plot: XGBoost model")
    #     plt.savefig(str(here('pipeline/graphing/graphs/skopt_graphs', f"{args.response}_convergence_plot.png")), dpi=300, bbox_inches='tight')

    #     #objective plot
    #     plot_objective(skopt_plot, size=2)
    #     plt.title(f"{args.response} Objective Plot: XGBoost model")
    #     plt.savefig(str(here('pipeline/graphing/graphs/skopt_graphs', f"{args.response}_objective_plot.png")), dpi=300, bbox_inches='tight')

    #     # evaluation plot
    #     plot_evaluations(skopt_plot, size=2)
    #     plt.title(f"{args.response} Evaluation Plot: XGBoost model")
    #     plt.savefig(str(here('pipeline/graphing/graphs/skopt_graphs', f"{args.response}_evaluation_plot.png")), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()