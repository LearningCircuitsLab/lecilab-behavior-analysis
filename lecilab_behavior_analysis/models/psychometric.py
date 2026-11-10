"""Psychometric curve: logistic with independent left and right lapses."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from lecilab_behavior_analysis.common import column_checker


# Define the lapse logistic function with independent lapses for left and right
def lapse_logistic_independent(params, x, y):
    lapse_left, lapse_right, beta, x0 = params
    # Ensure lapse rates are within [0, 0.5]
    lapse_left = np.clip(lapse_left, 0, 0.5)
    lapse_right = np.clip(lapse_right, 0, 0.5)
    # Predicted probabilities
    p_left = lapse_left + (1 - lapse_left - lapse_right) / (1 + np.exp(-beta * (x - x0)))
    # Negative log-likelihood
    nll = -np.sum(y * np.log(p_left) + (1 - y) * np.log(1 - p_left))
    return nll


def fit_lapse_logistic_independent(x, y, initial_params=None):
    """
    Fit a lapse logistic model with independent lapses for left and right choices.

    Args:
        x (np.array): Independent variable (e.g., stimulus intensity).
        y (np.array): Dependent variable (e.g., choice probabilities).
        initial_params (list, optional): Initial guess for the parameters [lapse_left, lapse_right, beta, x0].

    Returns:
        tuple: Fitted parameters and the minimized negative log-likelihood.
    """
    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values

    if initial_params is None:
        initial_params = [0.05, 0.05, 1, 0]  # Default initial values

    result = minimize(
        lapse_logistic_independent, initial_params, args=(x, y), bounds=[(0, 0.5), (0, 0.5), (None, None), (None, None)]
    )

    # Extract fitted parameters
    lapse_left, lapse_right, beta, x0 = result.x
    # print(f"Lapse Left: {lapse_left}, Lapse Right: {lapse_right}, Slope (Beta): {beta}, PSE (x0): {x0}")

    # Generate predictions
    xs = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    p_left = lapse_left + (1 - lapse_left - lapse_right) / (1 + np.exp(-beta * (xs - x0)))
    return p_left, (lapse_left, lapse_right, beta, x0)


def psychometric_curve_fitting_params(df: pd.DataFrame, x, y, valueType="discrete", bins=6):
    column_checker(df, required_columns={x, y})
    df_copy = df.copy(deep=True)
    if valueType == "discrete":
        if 0 in df_copy[x].unique():
            # if there are 0s in the data, we need to add a small value to avoid log(0)
            df_copy[x + "_fit"] = df_copy[x]
            df_copy[x + "_fit"].replace(0, 0.0001, inplace=True)
        df_copy[x + "_fit"] = np.sign(df_copy[x]) * (np.log(abs(df_copy[x])).round(4))
    else:
        # bin the continuous values when valueType is not discrete
        bin_groups = pd.cut(df_copy[x], bins=bins)
        labels = df_copy[x].groupby(bin_groups, observed=True).mean()
        df_copy[x + "_fit"] = pd.cut(df_copy[x], bins=bins, labels=labels).astype(float)
    p_left, params = fit_lapse_logistic_independent(df_copy[x + "_fit"], df_copy[y])
    return p_left, params
