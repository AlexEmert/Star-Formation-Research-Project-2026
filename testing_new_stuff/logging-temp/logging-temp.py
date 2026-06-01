from catboost import CatBoostRegressor
from xgboost import XGBRegressor, plot_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
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
import matplotlib.pyplot as plt

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
        
        for top, bottom in itertools.combinations(self.cols, 2):
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
            for top, bottom in itertools.combinations(self.cols, 2)
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
flux_cols = flux_cols[::-1]

log_y_train = np.log(y_train)

model = Pipeline([
    ('impute', 'passthrough'),
    ('ratio', RatioGenerator(cols=['F70', 'F24', 'F12', 'F8'])),
    ('scale', 'passthrough'),
    ('model', XGBRegressor(colsample_bytree=0.7826498515675833,
                            gamma=0.0796740126486606,
                            learning_rate=0.15085804664998495,
                            max_depth=3, 
                            min_child_weight=3,
                            n_estimators=644, 
                            n_jobs=-1,)
    )
])

model.fit(X_train, log_y_train)
rmse = root_mean_squared_error(y_test, np.exp(model.predict(X_test)))
r2 = r2_score(y_test, np.exp(model.predict(X_test)))
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")


xgb_model = model.named_steps['model']
importance_scores = model.named_steps['model'].feature_importances_
feature_names = model.named_steps['ratio'].get_feature_names_out(input_features=X_train.columns)

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance_scores
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
ax.set_title('XGBoost Feature Importances with log of TEMP as response')
ax.set_xlabel('Importance (Gain)')
ax.set_ylabel('Feature')
plt.tight_layout()
plt.savefig('xgb_log_temp_feature_importances_plot.png', dpi=300, bbox_inches='tight')
