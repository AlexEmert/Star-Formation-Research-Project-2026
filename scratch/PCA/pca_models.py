import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
from pyhere import here
from sklearn.preprocessing import StandardScaler, FunctionTransformer, RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from ratio_function import RatioGenerator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn import set_config
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import train_test_split

# make it work with data frames
set_config(transform_output="pandas")

phot = pd.read_csv(here("data/cleaned", "MIRION_cleaned_everything.csv"))

#temp test
phot_temp = phot.drop(columns=['YB', 'LRATIO', 'T_BOL', 'LM', 'L_BOL', 'MASS', 'DIAM', 'SURF_DENS'])

#diam test
phot_diam = phot.drop(columns=['YB', 'LRATIO', 'T_BOL', 'LM', 'L_BOL', 'MASS', 'TEMP', 'SURF_DENS'])

temp_X = phot_temp.drop(columns=['TEMP'])
temp_y = phot_temp['TEMP']

diam_X = phot_diam.drop(columns=['DIAM'])
diam_y = phot_diam['DIAM']

X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
    temp_X, temp_y,
    random_state=2026, 
    test_size = 0.2
)


X_train_diam, X_test_diam, y_train_diam, y_test_diam = train_test_split(
    diam_X, diam_y,
    random_state=2026, 
    test_size = 0.2
)

flux_cols = ['F8', 'F12', 'F24', 'F70', 'F160', 'F250', 'F350', 'F500', 'F870', 'F1100']
flux_cols = flux_cols[::-1]

pca_pipe = Pipeline([
    ('impute', SimpleImputer()),
    ('scale', StandardScaler()),
    ('pca', PCA())
])

# pca preprocessor
pca_preprocessor = ColumnTransformer([
    ('pca_pipe', pca_pipe, flux_cols)
], remainder='passthrough')

# Ratio processor
ratio_pipe = Pipeline([
        ('impute', SimpleImputer()),
        ('ratio', RatioGenerator(cols=flux_cols)),
        ('scale', StandardScaler())
])

#log ratio pipe
log_ratio_pipe = Pipeline([
        ('impute', SimpleImputer()),
        ('ratio', RatioGenerator(cols=flux_cols)),
        ('log', ColumnTransformer([('log_ratio', FunctionTransformer(func=np.log), ['F1100_over_F870', 'F1100_over_F500', 'F1100_over_F350', 'F1100_over_F250', 'F1100_over_F160', 'F1100_over_F70', 'F1100_over_F24', 'F1100_over_F12', 'F1100_over_F8', 'F870_over_F500', 'F870_over_F350', 'F870_over_F250', 'F870_over_F160', 'F870_over_F70', 'F870_over_F24', 'F870_over_F12', 'F870_over_F8', 'F500_over_F350', 'F500_over_F250', 'F500_over_F160', 'F500_over_F70', 'F500_over_F24', 'F500_over_F12', 'F500_over_F8', 'F350_over_F250', 'F350_over_F160', 'F350_over_F70', 'F350_over_F24', 'F350_over_F12', 'F350_over_F8', 'F250_over_F160', 'F250_over_F70', 'F250_over_F24', 'F250_over_F12', 'F250_over_F8', 'F160_over_F70', 'F160_over_F24', 'F160_over_F12', 'F160_over_F8', 'F70_over_F24', 'F70_over_F12', 'F70_over_F8', 'F24_over_F12', 'F24_over_F8', 'F12_over_F8'])], remainder='passthrough')),
        ('scale', StandardScaler())
])

rf_pca_pipe = Pipeline([
    ('pca', pca_preprocessor),
    ('model', RandomForestRegressor(random_state=2026, n_jobs=-1))
])

rf_ratio_pipe = Pipeline([
    ('ratio', ratio_pipe),
    ('model', RandomForestRegressor(random_state=2026, n_jobs=-1))
])

rf_log_ratio_pipe = Pipeline([
    ('logratio', log_ratio_pipe),
    ('model', RandomForestRegressor(random_state=2026, n_jobs=-1))
])

xgb_pca_pipe = Pipeline([
    ('pca', pca_preprocessor),
    ('model', XGBRegressor(random_state=2026, verbosity=0, n_jobs=-1))
])

xgb_ratio_pipe = Pipeline([
    ('ratio', ratio_pipe),
    ('model', XGBRegressor(random_state=2026, verbosity=0, n_jobs=-1))
])

xgb_log_ratio_pipe = Pipeline([
    ('logratio', log_ratio_pipe),
    ('model', XGBRegressor(random_state=2026, verbosity=0, n_jobs=-1))
])

rf_space = {
    'model__n_estimators': Integer(50, 500),
    'model__max_depth': Integer(3, 25),
    'model__min_samples_split': Integer(2, 15),
    'model__min_samples_leaf': Integer(1, 10),
    'model__max_features': Categorical(['sqrt', 'log2', 1.0]) 
}

xgb_space = {
    'model__n_estimators': Integer(50, 500),
    'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'model__max_depth': Integer(3, 10),
    'model__subsample': Real(0.5, 1.0),
    'model__colsample_bytree': Real(0.5, 1.0),
    'model__min_child_weight': Integer(1, 10)
}

rf_pca_search = BayesSearchCV(
    estimator=rf_pca_pipe,
    search_spaces=rf_space,
    n_iter=30,                         
    cv=5,                                 
    n_jobs=1,                           
    scoring='neg_mean_squared_error',       
    random_state=2026
)

rf_ratio_search = BayesSearchCV(
    estimator=rf_ratio_pipe,
    search_spaces=rf_space,
    n_iter=30,                         
    cv=5,                                 
    n_jobs=1,                           
    scoring='neg_mean_squared_error',       
    random_state=2026
)

rf_log_ratio_search = BayesSearchCV(
    estimator=rf_log_ratio_pipe,
    search_spaces=rf_space,
    n_iter=30,                         
    cv=5,                                 
    n_jobs=1,                           
    scoring='neg_mean_squared_error',       
    random_state=2026
)

xgb_pca_search = BayesSearchCV(
    estimator=xgb_pca_pipe,
    search_spaces=xgb_space,
    n_iter=30,                         
    cv=5,                                 
    n_jobs=1,                           
    scoring='neg_mean_squared_error',       
    random_state=2026
)

xgb_ratio_search = BayesSearchCV(
    estimator=xgb_ratio_pipe,
    search_spaces=xgb_space,
    n_iter=30,                         
    cv=5,                                 
    n_jobs=1,                           
    scoring='neg_mean_squared_error',       
    random_state=2026
)

xgb_log_ratio_search = BayesSearchCV(
    estimator=xgb_log_ratio_pipe,
    search_spaces=xgb_space,
    n_iter=30,                         
    cv=5,                                 
    n_jobs=1,                           
    scoring='neg_mean_squared_error',       
    random_state=2026
)


## yes I know that this could have been done much easier with nested for loops
## at first I planned on using a jupyter notebook but I quickly realized this would be more effective of a test
## but I didn't want to change all this so I just copy and pasted
## This test served purely to see whether or not PCA is more effective than ratios or log of ratios


# this may be the worst code I've ever written but it's too late to turn back now
# may have made this test a little more out of proportion than I expected
print("RF models:")
rf_pca_search.fit(X_train_temp, y_train_temp)
print("Best cv error for pca with temp:", -rf_pca_search.best_score_)
rf_pca_search.fit(X_train_diam, y_train_diam)
print("Best cv error for pca with diam:", -rf_pca_search.best_score_)
rf_ratio_search.fit(X_train_temp, y_train_temp)
print("Best cv error for ratio with temp:", -rf_ratio_search.best_score_)
rf_ratio_search.fit(X_train_diam, y_train_diam)
print("Best cv error for ratio with diam:", -rf_ratio_search.best_score_)
rf_log_ratio_search.fit(X_train_temp, y_train_temp)
print("Best cv error for log ratio with temp:", -rf_log_ratio_search.best_score_)
rf_log_ratio_search.fit(X_train_diam, y_train_diam)
print("Best cv error for log ratio with diam:", -rf_log_ratio_search.best_score_)
print("\n")
print("XGboost models:")
xgb_pca_search.fit(X_train_temp, y_train_temp)
print("Best cv error for pca with temp:", -xgb_pca_search.best_score_)
xgb_pca_search.fit(X_train_diam, y_train_diam)
print("Best cv error for pca with diam:", -xgb_pca_search.best_score_)
xgb_ratio_search.fit(X_train_temp, y_train_temp)
print("Best cv error for ratio with temp:", -xgb_ratio_search.best_score_)
xgb_ratio_search.fit(X_train_diam, y_train_diam)
print("Best cv error for ratio with diam:", -xgb_ratio_search.best_score_)
xgb_log_ratio_search.fit(X_train_temp, y_train_temp)
print("Best cv error for log ratio with temp:", -xgb_log_ratio_search.best_score_)
xgb_log_ratio_search.fit(X_train_diam, y_train_diam)
print("Best cv error for log ratio with diam:", -xgb_log_ratio_search.best_score_)
