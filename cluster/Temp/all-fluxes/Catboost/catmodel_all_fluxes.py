from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import itertools
from pyhere import here
from sklearn import set_config
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

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
    
phot = pd.read_csv(here("data/cleaned", "MIRION_cleaned_all_fluxes.csv"))
phot = phot.drop(columns=['LRATIO', 'T_BOL', 'LM', 'L_BOL', 'MASS', 'DIAM', 'SURF_DENS', 'YB'])

phot_X = phot.drop(columns=['TEMP'])
phot_y = phot['TEMP']

X_train, X_test, y_train, y_test = train_test_split(
    phot_X, 
    phot_y,
    train_size=0.8,
    random_state=2026
)

flux_cols = ['F8', 'F12', 'F24', 'F70', 'F160', 'F250', 'F350', 'F500', 'F870', 'F1100']

preproc = Pipeline([
    ('ratio', RatioGenerator(cols=flux_cols)),
    ('scale', StandardScaler())
])

catboost_pipe = Pipeline([
    ('preproc', preproc),
    ('model', CatBoostRegressor(random_state=2026))
])

search_spaces = {
    'model__iterations': Integer(200, 1000),
    'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'model__depth': Integer(3, 8), 
    'model__l2_leaf_reg': Real(1.0, 50.0, prior='log-uniform'),
    'model__random_strength': Real(1e-9, 10.0, prior='log-uniform'),
    'model__border_count': Integer(32, 255)
}

catboost_search = BayesSearchCV(
    catboost_pipe, 
    search_spaces,
    n_iter = 50,
    n_jobs = -1,
    cv = 5,
    random_state = 2026,
    scoring='neg_root_mean_squared_error'
)

catboost_search.fit(X_train, y_train)

print(f"Temp: Best catboost score with all fluxes, ratio and scale only: {-catboost_search.best_score_}")