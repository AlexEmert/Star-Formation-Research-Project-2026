import pandas as pd
import numpy as np
import itertools
from pyhere import here
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config
from skopt.space import Real, Integer, Categorical

# make it work with data frames
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
        # Create a copy to avoid SettingWithCopy warnings or mutating the original
        X_out = X.copy()
        
        for top, bottom in itertools.combinations(self.cols, 2):
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
            for top, bottom in itertools.combinations(self.cols, 2)
        ]

        # IMPORTANT: include ALL original input features
        return np.array(input_features + ratio_features, dtype=object)
    


class LogRatioGenerator(BaseEstimator, TransformerMixin):
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
        # Create a copy to avoid SettingWithCopy warnings or mutating the original
        X_out = X.copy()
        
        for top, bottom in itertools.combinations(self.cols, 2):
            new_col_name = f"log_{top}_over_{bottom}"
            X_out[new_col_name] = np.log(X_out[top] / X_out[bottom])
            
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
            f"log_{top}_over_{bottom}"
            for top, bottom in itertools.combinations(self.cols, 2)
        ]

        # IMPORTANT: include ALL original input features
        return np.array(input_features + ratio_features, dtype=object)
    

def check_and_expand_space(best_params, current_space, definitive_bounds, tolerance=0.05, expansion=0.5):
    """
    Checks if best_params are near the bounds of the current_space.
    Returns a boolean indicating if expansion happened, and the new search space.
    """
    new_space = {}
    bounds_expanded = False
    
    for param_name, skopt_obj in current_space.items():
        best_val = best_params[param_name]
        # declare possible bounds
        if isinstance(skopt_obj, Real) or isinstance(skopt_obj, Integer):
            abs_low, abs_high = definitive_bounds[param_name]

        # Real Parameters
        if isinstance(skopt_obj, Real):
            # declare current bounds
            low, high = skopt_obj.bounds
            span = high - low
            
            if skopt_obj.prior == "uniform":
                if best_val <= low + (span * tolerance):
                    new_low = low - (span * expansion)
                    bounds_expanded = True
                    if new_low < abs_low:
                        new_low = low
                        bounds_expanded = False
                    new_space[param_name] = Real(new_low, high, prior="uniform")
                    
                    
                elif best_val >= high - (span * tolerance):
                    new_high = high + (span * expansion)
                    bounds_expanded = True
                    if new_high > abs_high:
                        new_high = high
                        bounds_expanded = False
                    new_space[param_name] = Real(low, new_high, prior="uniform")
                    
                    
                else:
                    new_space[param_name] = skopt_obj # Unchanged

            if skopt_obj.prior == "log-uniform":
                log_low, log_high = np.log10(low), np.log10(high)
                log_span = log_high - log_low
                log_best = np.log10(best_val)
                
                if log_best <= log_low + (log_span * tolerance):
                    new_log_low = log_low - (log_span * expansion)
                    new_low = 10 ** new_log_low
                    new_low = max(new_low, abs_low)
                    new_space[param_name] = Real(new_low, high, prior='log-uniform')
                    bounds_expanded = True
                    
                elif log_best >= log_high - (log_span * tolerance):
                    new_log_high = log_high + (log_span * expansion)
                    new_high = 10 ** new_log_high
                    new_high = min(new_high, abs_high)
                    new_space[param_name] = Real(low, new_high, prior='log-uniform')
                    bounds_expanded = True
                    
                else:
                    new_space[param_name] = skopt_obj

        # Integer Parameters
        elif isinstance(skopt_obj, Integer):
            low, high = skopt_obj.bounds
            span = high - low
            # Use max(1, ...) to ensure we always tolerate/expand by at least 1 unit
            tol_val = max(1, int(span * tolerance))
            exp_val = max(1, int(span * expansion))
            
            if best_val <= low + tol_val:
                new_low = low - exp_val
                if new_low < abs_low:
                    new_low = low
                new_space[param_name] = Integer(new_low, high, prior=skopt_obj.prior)
                bounds_expanded = True
                
            elif best_val >= high - tol_val:
                new_high = high + exp_val
                if new_high > abs_high:
                    new_high = high
                new_space[param_name] = Integer(low, new_high, prior=skopt_obj.prior)
                bounds_expanded = True
                
            else:
                new_space[param_name] = skopt_obj # Unchanged
                
        # 3. Handle Categorical Parameters (Cannot be expanded)
        else:
            new_space[param_name] = skopt_obj
            
    return bounds_expanded, new_space

def pinball_loss_function(y_true, y_pred, alphas=(0.025, 0.5, 0.975)):
    '''custom loss function for multiple quantile regression that scores based on quantiles'''
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Expand y_true to shape (n_samples, 1) so it broadcasts 
    # correctly against y_pred of shape (n_samples, n_quantiles)
    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
        
    residuals = y_true - y_pred
    alphas = np.array(alphas)
    
    # Apply the pinball loss formula across all quantiles simultaneously
    loss = np.maximum(alphas * residuals, (alphas - 1.0) * residuals)
    
    # Return the mean loss across both samples and quantiles
    return np.mean(loss)