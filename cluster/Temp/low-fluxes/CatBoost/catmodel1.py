from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from cluster.ratio_function import RatioGenerator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import itertools
from pyhere import here
from sklearn import set_config

# Force all scikit-learn transformers to output pandas DataFrames
set_config(transform_output="pandas")


class RatioGenerator(BaseEstimator, TransformerMixin):
    '''
    A custom transformer that generates new features by taking the ratios of all combinations of specified columns.
    For use with the flux columns
    '''
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        # add a dummy attribute so sklearn knows this transformer is fitted
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame")
        
        # Create a copy to avoid SettingWithCopy warnings or mutating the original
        X_out = X.copy()
        
        for top, bottom in itertools.permutations(self.cols, 2):
            new_col_name = f"{top}_over_{bottom}"
            X_out[new_col_name] = X_out[top] / (X_out[bottom] + 1e-8 ) #add epsilon to reduce division by 0 errors
            
        return X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided")

        input_features = list(input_features)

        # Validate columns exist
        missing = set(self.cols) - set(input_features)
        if missing:
            raise ValueError(f"Missing columns in input_features: {missing}")

        # Generate ratio feature names
        ratio_features = [
            f"{top}_over_{bottom}"
            for top, bottom in itertools.permutations(self.cols, 2)
        ]

        # IMPORTANT: include ALL original input features
        return np.array(input_features + ratio_features, dtype=object)


phot = pd.read_csv(here("data/cleaned", "MIRION_cleaned_low_fluxes.csv"))
phot = phot.drop(columns=['LRATIO', 'T_BOL', 'LM', 'L_BOL', 'MASS', 'DIAM', 'SURF_DENS', 'YB'])

phot_X = phot.drop(columns=['TEMP'])
phot_y = phot['TEMP']

X_train, X_test, y_train, y_test = train_test_split(
    phot_X, 
    phot_y,
    train_size=0.8,
    random_state=2026
)

flux_cols = ['F8', 'F12', 'F24', 'F70']

preproc1 = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('ratio', RatioGenerator(cols=flux_cols)),
    ('impute_check', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

preproc2 = Pipeline([
    ('impute', SimpleImputer(strategy='median', add_indicator=True)),
    ('ratio', RatioGenerator(cols=flux_cols)),
    ('impute_check', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

preproc3 = Pipeline([
    ('ratio', RatioGenerator(cols=flux_cols)),
    ('scale', StandardScaler())
])

catboost_pipe1 = Pipeline([
    ('preproc', preproc1),
    ('model', CatBoostRegressor(random_state=2026, verbose=False))
])

catboost_pipe2 = Pipeline([
    ('preproc', preproc2),
    ('model', CatBoostRegressor(random_state=2026, verbose=False))
])

catboost_pipe3 = Pipeline([
    ('preproc', preproc3),
    ('model', CatBoostRegressor(random_state=2026, verbose=False))
])

catboost_search1 = GridSearchCV(
    catboost_pipe1,
    param_grid={
        'model__iterations': [100, 200],
        'model__learning_rate': [0.01, 0.1],
        'model__depth': [4, 6]
    },
    cv=5,
    n_jobs=-1
)

catboost_search2 = GridSearchCV(
    catboost_pipe2,
    param_grid={
        'model__iterations': [100, 200],
        'model__learning_rate': [0.01, 0.1],
        'model__depth': [4, 6]
    },
    cv=5,
    n_jobs=-1
)

catboost_search3 = GridSearchCV(
    catboost_pipe3,
    param_grid={
        'model__iterations': [100, 200],
        'model__learning_rate': [0.01, 0.1],
        'model__depth': [4, 6]
    },
    cv=5,
    n_jobs=-1
)

catboost_search1.fit(X_train, y_train)
catboost_search2.fit(X_train, y_train)
catboost_search3.fit(X_train, y_train)

print("Best score for catboost with impute, without indicator:", catboost_search1.best_score_)
print("Best score for catboost with impute and indicator:", catboost_search2.best_score_)
print("Best score for catboost with only ratio and scale:", catboost_search3.best_score_)

print("Best parameters for catboost with impute, without indicator:", catboost_search1.best_params_)
print("Best parameters for catboost with impute and indicator:", catboost_search2.best_params_)
print("Best parameters for catboost with only ratio and scale:", catboost_search3.best_params_)