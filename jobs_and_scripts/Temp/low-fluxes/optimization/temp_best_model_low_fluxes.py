from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
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

pipe = Pipeline([
    ('impute', 'passthrough'),
    ('ratio', RatioGenerator(cols=flux_cols)),
    ('scale', 'passthrough'),
    ('model', XGBRegressor())
])

# search_space = [
#     {
#     'impute': [SimpleImputer(strategy='median')],
#     'scale': [StandardScaler()],
#     'model': XGBRegressor(random_state=2026),
#     'model__n_estimators': Integer(100, 800),
#     'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
#     'model__max_depth': Integer(3, 9),
#     'model__min_child_weight': Integer(1, 7),
#     'model__subsample': Real(0.6, 1.0),
#     'model__colsample_bytree': Real(0.6, 1.0),
#     'model__gamma': Real(0.0, 5.0),
#     'model__reg_alpha': Real(1e-4, 10.0, prior='log-uniform'),
#     'model__reg_lambda': Real(1e-4, 10.0, prior='log-uniform')
#     },
#     {}
# ]

search_space = {
    'impute': [SimpleImputer(strategy='median')],
    'scale': [StandardScaler()],
    'model': RandomForestRegressor(random_state=2026),
    'model__n_estimators': Integer(100, 800),
    'model__max_depth': Integer(3, 20),
    'model__min_samples_split': Integer(2, 10),
    'model__max_features': Categorical(['sqrt', 'log2', None])
}

opt = BayesSearchCV(
    pipe, 
    search_space,
    n_iter=30,
    cv=5,
    scoring='neg_root_mean_squared_error',
    random_state=2026,
    n_jobs=-1
)

opt.fit(X_train, y_train)
print("Best RMSE for temp low fluxes random forest optimization trial 1: ", -opt.best_score_)