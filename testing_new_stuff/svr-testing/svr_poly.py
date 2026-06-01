from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import itertools
from pyhere import here
from sklearn import set_config
from sklearn.metrics import r2_score, root_mean_squared_error

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

models = {
    "SVR (poly)": {
        "pipe": Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('ratio', RatioGenerator(cols=flux_cols)),
            ('scale', RobustScaler()),
            ('model', SVR())
        ]),
        "space": {
            'scale': Categorical([StandardScaler(), RobustScaler()]),
            'model__kernel': Categorical(['poly']),
            'model__C': Real(0.1, 100, prior='log-uniform'),
            'model__gamma': Real(1e-4, 1e+1, prior='log-uniform'),
            'model__epsilon': Real(0.01, 1.0, prior='log-uniform'),
            'model__coef0': Real(0, 10),
            'model__degree': Integer(1, 5)
        }
    }
}

best_overall_score = float('-inf')
best_overall_model = None

for name, setup in models.items():
    print(f"--- Running {name} ---")
    
    opt = BayesSearchCV(
        estimator=setup["pipe"],
        search_spaces=setup["space"],
        n_iter=30, 
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=1,
        random_state=2026
    )

    opt.fit(X_train, y_train)

    print(f"Best Score: {-opt.best_score_:.4f}")
    print(f"Best Params: {dict(opt.best_params_)}\n")

    # Keep track of the absolute winner
    if opt.best_score_ > best_overall_score:
        best_overall_score = -opt.best_score_
        best_overall_model = opt.best_estimator_

# print(f"*** Best Temp low fluxes results ***\nScore: {best_overall_score}\nModel: {best_overall_model}")

# best_overall_model.fit(X_train, y_train)
# y_pred = best_overall_model.predict(X_test)
# best_mod_rmse = root_mean_squared_error(y_test, y_pred)
# best_mod_r2 = r2_score(y_test, y_pred)
# print(f"Best overall model performance on test set:\nRMSE: {best_mod_rmse:.4f}\nR^2: {best_mod_r2:.4f}")

