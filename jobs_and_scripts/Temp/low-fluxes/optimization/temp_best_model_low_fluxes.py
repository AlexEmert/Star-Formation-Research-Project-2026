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
    "CatBoost (Imputed)": {
        "pipe": Pipeline([
            ('impute', SimpleImputer()),
            ('ratio', RatioGenerator(cols=flux_cols)),
            ('scale', RobustScaler()),
            ('model', CatBoostRegressor(random_state=2026, verbose=0, thread_count=-1))
        ]),
        "space": {
            'impute__strategy': Categorical(['mean', 'median']),
            'scale': Categorical([StandardScaler(), RobustScaler(), 'passthrough']),
            'model__iterations': Integer(100, 1000),
            'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'model__depth': Integer(4, 10),
            'model__l2_leaf_reg': Real(1, 10, prior='uniform'),
            'model__random_strength': Real(1e-9, 10, prior='log-uniform'),
            'model__bagging_temperature': Real(0.0, 1.0)
        }
    },
    "Catboost (No Impute)": {
        "pipe": Pipeline([
            ('impute', "passthrough"),
            ('ratio', RatioGenerator(cols=flux_cols)),
            ('scale', RobustScaler()),
            ('model', CatBoostRegressor(random_state=2026, verbose=0, thread_count=-1))
        ]),
        "space": {
            'scale': Categorical([StandardScaler(), RobustScaler(), 'passthrough']),
            'model__iterations': Integer(100, 1000),
            'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'model__depth': Integer(4, 10),
            'model__l2_leaf_reg': Real(1, 10, prior='uniform'),
            'model__random_strength': Real(1e-9, 10, prior='log-uniform'),
            'model__bagging_temperature': Real(0.0, 1.0)
        }
    },
    "XGBoost (Imputed)": {
        "pipe": Pipeline([
            ('impute', SimpleImputer()),
            ('ratio', RatioGenerator(cols=flux_cols)),
            ('scale', RobustScaler()),
            ('model', XGBRegressor(random_state=2026, verbosity=0, n_jobs=-1))
        ]),
        "space": {
            'impute__strategy': Categorical(['mean', 'median']),
            'scale': Categorical([StandardScaler(), RobustScaler(), 'passthrough']),
            'model__n_estimators': Integer(100, 800),
            'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'model__max_depth': Integer(3, 9),
            'model__min_child_weight': Integer(1, 7),
            'model__subsample': Real(0.6, 1.0),
            'model__colsample_bytree': Real(0.6, 1.0),
            'model__gamma': Real(0.0, 5.0),
            'model__reg_alpha': Real(1e-4, 10.0, prior='log-uniform'),
            'model__reg_lambda': Real(1e-4, 10.0, prior='log-uniform')
    }
    },
    "XGBoost (No Impute)": {
        "pipe": Pipeline([
            ('impute', "passthrough"),
            ('ratio', RatioGenerator(cols=flux_cols)),
            ('scale', RobustScaler()),
            ('model', XGBRegressor(random_state=2026, verbosity=0, n_jobs=-1))
        ]),
        "space": {
            'scale': Categorical([StandardScaler(), RobustScaler(), 'passthrough']),
            'model__n_estimators': Integer(100, 800),
            'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'model__max_depth': Integer(3, 9),
            'model__min_child_weight': Integer(1, 7),
            'model__subsample': Real(0.6, 1.0),
            'model__colsample_bytree': Real(0.6, 1.0),
            'model__gamma': Real(0.0, 5.0),
            'model__reg_alpha': Real(1e-4, 10.0, prior='log-uniform'),
            'model__reg_lambda': Real(1e-4, 10.0, prior='log-uniform')
            }     
    },
    "Random Forest": {
        "pipe": Pipeline([
            ('impute', SimpleImputer()),
            ('ratio', RatioGenerator(cols=flux_cols)),
            ('scale', RobustScaler()),
            ('model', RandomForestRegressor(random_state=2026, verbose=0, n_jobs=-1))
        ]),
        "space": {
            'impute__strategy': Categorical(['mean', 'median']),
            'scale': Categorical([StandardScaler(), RobustScaler(), 'passthrough']),
            'model__n_estimators': Integer(100, 800),
            'model__max_depth': Integer(5, 50),
            'model__min_samples_split': Integer(2, 20),
            'model__min_samples_leaf': Integer(1, 20),
            'model__max_features': Categorical(['sqrt', 'log2', None]), 
            'model__bootstrap': Categorical([True, False])
        }
    },
    "Decision Tree": {
        "pipe": Pipeline([
            ('impute', SimpleImputer()),
            ('ratio', RatioGenerator(cols=flux_cols)),
            ('scale', RobustScaler()),
            ('model', DecisionTreeRegressor(random_state=2026))
        ]),
        "space": {
            'impute__strategy': Categorical(['mean', 'median']),
            'scale': Categorical([StandardScaler(), RobustScaler(), 'passthrough']),
            'model__max_depth': Integer(5, 50),
            'model__min_samples_split': Integer(2, 20),
            'model__min_samples_leaf': Integer(1, 20),
            'model__max_features': Categorical(['sqrt', 'log2', None])
        }
    },
    "SVR (rbf)": {
        "pipe": Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('ratio', RatioGenerator(cols=flux_cols)),
            ('scale', RobustScaler()),
            ('model', SVR())
        ]),
        "space": {
            'scale': Categorical([StandardScaler(), RobustScaler()]),
            'model__kernel': Categorical(['rbf']),
            'model__C': Real(0.1, 100, prior='log-uniform'),
            'model__gamma': Real(1e-4, 1e+1, prior='log-uniform'),
            'model__epsilon': Real(0.01, 1.0, prior='log-uniform')
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

best_overall_model.fit(X_train, y_train)
y_pred = best_overall_model.predict(X_test)
best_mod_rmse = root_mean_squared_error(y_test, y_pred)
best_mod_r2 = r2_score(y_test, y_pred)
print(f"Best overall model performance on test set:\nRMSE: {best_mod_rmse:.4f}\nR^2: {best_mod_r2:.4f}")


## Try again, but logging the response variable to see what happens
log_phot_y = np.log1p(phot['TEMP'])

new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(
    phot_X, 
    log_phot_y,
    train_size=0.8,
    random_state=2026
)

best_overall_log_score = float('-inf')
best_overall_log_model = None

for name, setup in models.items():
    print(f"--- Running {name} for log of temp---")
    
    log_opt = BayesSearchCV(
        estimator=setup["pipe"],
        search_spaces=setup["space"],
        n_iter=30, 
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=1,
        random_state=2026
    )

    log_opt.fit(new_X_train, new_y_train)

    print(f"Best Score with log temp (not translatable): {-log_opt.best_score_:.4f}")
    print(f"Best Params with log temp (not translatable): {dict(log_opt.best_params_)}\n")

    # Keep track of the absolute winner
    if log_opt.best_score_ > best_overall_log_score:
        best_overall_log_score = -log_opt.best_score_
        best_overall_log_model = log_opt.best_estimator_

# print(f"*** Best Temp low fluxes results ***\nScore: {best_overall_log_score}\nModel: {best_overall_log_model}")

best_overall_log_model.fit(new_X_train, new_y_train)
new_y_pred = best_overall_log_model.predict(new_X_test)
best_log_mod_rmse = root_mean_squared_error(new_y_test, new_y_pred)
best_log_mod_r2 = r2_score(new_y_test, new_y_pred)
print(f"Best overall model performance on test set:\nRMSE: {best_log_mod_rmse:.4f}\nR^2: {best_log_mod_r2:.4f}")