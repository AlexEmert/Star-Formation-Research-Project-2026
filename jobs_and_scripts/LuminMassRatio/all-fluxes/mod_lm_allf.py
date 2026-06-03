from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
import itertools
from pyhere import here
from sklearn import set_config
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.metrics import r2_score, root_mean_squared_error
from skopt.plots import plot_convergence, plot_objective, plot_evaluations
import matplotlib.pyplot as plt


# Force all scikit-learn transformers to output pandas DataFrames
set_config(transform_output="pandas")


# Custom function that generates all unique ratios
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


# read data with all response variables and everything else
phot = pd.read_csv(here("data/cleaned", "MIRION_cleaned_everything.csv"))

#drop the other response variables so they don't interact with the predictions
phot = phot.drop(columns=['LRATIO', 'TEMP', 'MASS', 'L_BOL', 'SURF_DENS', 'DIAM', 'T_BOL', 'YB'])

phot_X = phot.drop(columns=['LM'])
phot_y = phot['LM']

X_train, X_test, y_train, y_test = train_test_split(
    phot_X, 
    phot_y,
    train_size=0.8,
    random_state=2026
)

flux_cols = ['F8', 'F12', 'F24', 'F70', 'F160', 'F250', 'F350', 'F500', 'F870', 'F1100']
flux_cols = flux_cols [::-1] # reverse the order to make ratios large/small which is what physicists use

models = {
    "CatBoost": {
        "pipe": Pipeline([
            ('impute', SimpleImputer()),
            ('ratio', RatioGenerator(cols=flux_cols)),
            ('scale', RobustScaler()),
            ('model', CatBoostRegressor(random_state=2026, verbose=0, thread_count=-1, loss_function='RMSE'))
        ]),
        "space": {
            'impute': Categorical([SimpleImputer(strategy='mean'), SimpleImputer(strategy='median'), KNNImputer(), 'passthrough']),
            'scale': Categorical([StandardScaler(), RobustScaler(), 'passthrough']),
            "model__iterations": Integer(100, 3000),
            "model__learning_rate": Real(1e-4, 0.5, prior="log-uniform"),
            "model__depth": Integer(3, 12),
            "model__l2_leaf_reg": Real(1e-3, 100.0, prior="log-uniform"),
            "model__random_strength": Real(1e-9, 10.0, prior="log-uniform"),
            "model__bagging_temperature": Real(0.0, 10.0, prior="uniform"),
            "model__border_count": Integer(32, 255),
            "model__min_data_in_leaf": Integer(1, 100),
            "model__colsample_bylevel": Real(0.05, 1.0, prior="uniform"),
            "model__grow_policy": Categorical(["SymmetricTree", "Depthwise", "Lossguide"])
        }       
    },
    "XGBoost": {
        "pipe": Pipeline([
            ('impute', SimpleImputer()),
            ('ratio', RatioGenerator(cols=flux_cols)),
            ('scale', RobustScaler()),
            ('model', XGBRegressor(random_state=2026, verbosity=0, n_jobs=-1))
        ]),
        "space": {
            'impute': Categorical([SimpleImputer(strategy='mean'), SimpleImputer(strategy='median'), KNNImputer(), 'passthrough']),
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
    },
    "Random Forest": {
        "pipe": Pipeline([
            ('impute', SimpleImputer()),
            ('ratio', RatioGenerator(cols=flux_cols)),
            ('scale', RobustScaler()),
            ('model', RandomForestRegressor(random_state=2026, verbose=0, n_jobs=-1))
        ]),
        "space": {
            'impute': Categorical([SimpleImputer(strategy='mean'), SimpleImputer(strategy='median'), KNNImputer()]),
            'scale': Categorical([StandardScaler(), RobustScaler(), 'passthrough']),
            "model__n_estimators": Integer(10, 1000),
            "model__max_depth": Integer(3, 30),
            "model__min_samples_split": Integer(2, 20),
            "model__min_samples_leaf": Integer(1, 20),
            "model__max_features": Categorical(["sqrt", "log2", None]), 
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
            'impute': Categorical([SimpleImputer(strategy='mean'), SimpleImputer(strategy='median'), KNNImputer()]),
            'scale': Categorical([StandardScaler(), RobustScaler(), 'passthrough']),
            "model__max_depth": Integer(3, 30),
            "model__min_samples_split": Integer(2, 20),
            "model__min_samples_leaf": Integer(1, 20),
            "model__max_features": Categorical(["sqrt", "log2", None])
        }
    }
    #   leaving out SVR because it is not only slow but consistently underperforms the other models
    # ,
    # "SVR (rbf)": {
    #     "pipe": Pipeline([
    #         ('impute', SimpleImputer(strategy='median')),
    #         ('ratio', RatioGenerator(cols=flux_cols)),
    #         ('scale', RobustScaler()),
    #         ('model', SVR())
    #     ]),
    #     "space": {
    #         'scale': Categorical([StandardScaler(), RobustScaler()]),
    #         'model__kernel': Categorical(['rbf']),
    #         'model__C': Real(0.1, 100, prior='log-uniform'),
    #         'model__gamma': Real(1e-4, 1e+1, prior='log-uniform'),
    #         'model__epsilon': Real(0.01, 1.0, prior='log-uniform')
    #     }
    # }
}


# keep track of best score/model to test on the test set
best_overall_score = float('inf')
best_overall_model = None
global_best_result = None
best_model_name = ""

# iterate through different search spaces/models
for name, setup in models.items():
    print(f"{name}:")
    
    opt = BayesSearchCV(
        estimator=setup["pipe"],
        search_spaces=setup["space"],
        n_iter=45, 
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=1,
        random_state=2026
    )

    opt.fit(X_train, y_train)

    print(f"Best CV Score: {-opt.best_score_:.4f}")
    # print(f"Best Params: {dict(opt.best_params_)}\n") don't need to print best params for all models

    # Keep track of the best model
    if -opt.best_score_ < best_overall_score:
        best_overall_score = -opt.best_score_
        best_overall_model = opt.best_estimator_
        global_best_result = opt.optimizer_results_[0]
        best_model_name = name

# calculate metrics only on best model to prevent data leakage
best_overall_model.fit(X_train, y_train)
y_pred = best_overall_model.predict(X_test)
best_mod_rmse = root_mean_squared_error(y_test, y_pred)
best_mod_r2 = r2_score(y_test, y_pred)


print("\n")
## Try again, but logging the response variable incase that is more important
log_phot_y = np.log(phot_y)

new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(
    phot_X, 
    log_phot_y,
    train_size=0.8,
    random_state=2026
)

best_overall_log_score = float('inf')
best_overall_log_model = None
global_best_log_result = None
best_log_model_name = ""

for name, setup in models.items():
    print(f"{name} for log of lm:")
    
    log_opt = BayesSearchCV(
        estimator=setup["pipe"],
        search_spaces=setup["space"],
        n_iter=200, 
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=1,
        random_state=2026
    )

    log_opt.fit(new_X_train, new_y_train)

    print(f"Best CV Score: {-log_opt.best_score_:.4f}")
    # print(f"Best Params: {dict(log_opt.best_params_)}\n") don't need to print best params for all models

    # Keep track of the best model with log response
    if -log_opt.best_score_ < best_overall_log_score:
        best_overall_log_score = -log_opt.best_score_
        best_overall_log_model = log_opt.best_estimator_
        global_best_log_result = log_opt.optimizer_results_[0]
        best_log_model_name = name

best_overall_log_model.fit(new_X_train, new_y_train)
new_y_pred = best_overall_log_model.predict(new_X_test)
best_log_mod_rmse = root_mean_squared_error(new_y_test, new_y_pred)
best_log_mod_r2 = r2_score(new_y_test, new_y_pred)

print("\n")
print("Summary of Best Models:")
print(f"Best overall CV RMSE: {best_overall_score:.4f}\n")
print(f"Best overall model performance on test set:\nRMSE: {best_mod_rmse:.4f}\nR^2: {best_mod_r2:.4f}\n")
print(f"Final model impute strategy: {best_overall_model.named_steps['impute']}")
print(f"Final model scaler: {best_overall_model.named_steps['scale']}")
print(f"Final model parameters: {best_overall_model.named_steps['model']}")


print(f"Best overall log CV RMSE: {best_overall_log_score:.4f}\n")
print(f"Best overall log model performance on test set:\nRMSE: {best_log_mod_rmse:.4f}\nR^2: {best_log_mod_r2:.4f}\n")
print(f"Final log model impute strategy: {best_overall_log_model.named_steps['impute']}")
print(f"Final log model scaler: {best_overall_log_model.named_steps['scale']}")
print(f"Final log model parameters: {best_overall_log_model.named_steps['model']}")



## graphing the skopt plots for the best model

# first convergence plot
plt.figure(figsize=(8, 6))
plot_convergence(global_best_result)
plt.title(f"Convergence Plot: {best_model_name}")
plt.savefig(f"lm_convergence_plot.png", dpi=300, bbox_inches='tight')

#objective plot
plot_objective(global_best_result, size=2)
plt.title(f"Objective Plot: {best_model_name}")
plt.savefig(f"lm_objective_plot.png", dpi=300, bbox_inches='tight')

# evaluation plot
plot_evaluations(global_best_result, size=2)
plt.title(f"Evaluation Plot: {best_model_name}")
plt.savefig(f"lm_evaluation_plot.png", dpi=300, bbox_inches='tight')


## log response variable skopt plots
plt.figure(figsize=(8, 6))
plot_convergence(global_best_log_result)
plt.title(f"Convergence Plot: {best_log_model_name}")
plt.savefig(f"log_lm_convergence_plot.png", dpi=300, bbox_inches='tight')

#objective plot
plot_objective(global_best_log_result, size=2)
plt.title(f"Objective Plot: {best_log_model_name}")
plt.savefig(f"log_lm_objective_plot_.png", dpi=300, bbox_inches='tight')

# evaluation plot
plot_evaluations(global_best_log_result, size=2)
plt.title(f"Evaluation Plot: {best_log_model_name}")
plt.savefig(f"log_lm_evaluation_plot.png", dpi=300, bbox_inches='tight')
