import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr, kendalltau
import torch

def to_numpy(data) -> np.ndarray:
    """
    Robustly converts various data types to a 1D numpy.ndarray.
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.to_numpy()
    elif not isinstance(data, np.ndarray):
        try:
            data = np.array(data)
        except Exception as e:
            raise TypeError(f"Cannot convert type {type(data)} to numpy.ndarray: {e}")
    
    return data.flatten() # Ensure the output is a 1D array

def regression_metrics(y_true, y_pred, just_main_metric=False):
    """
    Calculates a suite of regression metrics, handling potential non-finite values.
    """
    y_true_np = to_numpy(y_true)
    y_pred_np = to_numpy(y_pred)

    # --- Important: Filter out any non-finite values (NaN, Inf) before calculation ---
    finite_mask = np.isfinite(y_true_np) & np.isfinite(y_pred_np)
    
    if np.sum(finite_mask) < 2: # Need at least 2 points to calculate correlations and R2
        # print("Warning: Not enough finite data points to calculate metrics.")
        default_metrics = {'rmse': float('nan'), 'r2': float('nan'), 'mse': float('nan'), 'mae': float('nan'),
                           'pr': float('nan'), 'sr': float('nan'), 'kr': float('nan')}
        return default_metrics if not just_main_metric else {'rmse': float('nan')}

    y_true_f = y_true_np[finite_mask]
    y_pred_f = y_pred_np[finite_mask]

    # --- Use scikit-learn's robust implementations ---
    try:
        mae = mean_absolute_error(y_true_f, y_pred_f)
        mse = mean_squared_error(y_true_f, y_pred_f)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_f, y_pred_f)
    except Exception:
        mae, mse, rmse, r2 = float('nan'), float('nan'), float('nan'), float('nan')

    if just_main_metric:
        return {'rmse': rmse}

    # --- Correlation coefficients with safety checks ---
    try:
        pearson_r, _ = pearsonr(y_true_f, y_pred_f)
    except (ValueError, FloatingPointError):
        pearson_r = float('nan')
        
    try:
        spearman_r, _ = spearmanr(y_true_f, y_pred_f)
    except (ValueError, FloatingPointError):
        spearman_r = float('nan')
        
    try:
        kendall_r, _ = kendalltau(y_true_f, y_pred_f)
    except (ValueError, FloatingPointError):
        kendall_r = float('nan')

    metrics_dict = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'pr': pearson_r, # Pearson correlation
        'sr': spearman_r, # Spearman correlation
        'kr': kendall_r   # Kendall Tau correlation
    }
    
    return metrics_dict