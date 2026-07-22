import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import itertools
from pyhere import here
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import cross_val_score
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
            X_out[new_col_name] = X_out[top] / X_out[bottom]
            
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
    ('impute', SimpleImputer(strategy='median')),
    ('ratio', RatioGenerator(cols=flux_cols)),
    ('impute_check', SimpleImputer(strategy='median')),
    ('polynomials', ColumnTransformer([('polynomial', PolynomialFeatures(degree=2), flux_cols)])),
    ('scale', StandardScaler())
])

## setting up combinations of linear regression, lasso, and preprocessing

lr_no_poly = Pipeline([
    ('preproc', preproc1),
    ('model', LinearRegression())
])

lr_no_poly_score = cross_val_score(
    estimator=lr_no_poly, 
    X=X_train, 
    y=y_train, 
    cv=5,
    scoring='neg_mean_squared_error'
)

lr_no_poly_rmse = np.sqrt(-lr_no_poly_score).mean()

# Linear Regression with polynomial term
lr_with_poly = Pipeline([
    ('preproc', preproc2),
    ('model', LinearRegression())
])

lr_with_poly_score = cross_val_score(lr_with_poly, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
lr_with_poly_rmse = np.sqrt(-lr_with_poly_score).mean()


# Lasso pipeline to use with LassoCV
lasso_pipe = Pipeline([
    ('preproc', preproc2),
    ('model', LassoCV(cv=5, n_jobs=-1))
])

lasso_pipe.fit(X_train, y_train)
best_mse = lasso_pipe.named_steps['model'].mse_path_[lasso_pipe.named_steps['model'].alphas_ == lasso_pipe.named_steps['model'].alpha_].mean()
lasso_best_rmse = np.sqrt(best_mse)

feature_names = lasso_pipe.named_steps['preproc'].get_feature_names_out()
coef_df = pd.DataFrame({'feature': feature_names, 'coef': lasso_pipe.named_steps['model'].coef_})
selected = coef_df[coef_df['coef'] != 0]


print(f"Simple Linear Regression RMSE: {lr_no_poly_rmse}")
print(f"Linear Regression with polynomials RMSE: {lr_with_poly_rmse}")
print(f"LASSO best rmse: {lasso_best_rmse}")
print(selected.head(10))