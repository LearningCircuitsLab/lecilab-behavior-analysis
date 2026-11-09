"""Logistic regressions of the choices and how much each regressor contributes."""
from collections import defaultdict
from sklearn.metrics import r2_score
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm

from lecilab_behavior_analysis.common import column_checker
from lecilab_behavior_analysis.metrics.choices import get_time_kernel_impact, previous_impact_on_time_kernel
from lecilab_behavior_analysis.metrics.trial_events import transform_side_choice_to_numeric


def logi_model_fit_input(df: pd.DataFrame, X, y, method='newton'):
    column_checker(df, {x for x in X})
    column_checker(df, {y})

    # drop NaN values if any
    df_for_fit = df.dropna(subset=X + [y])
    df_for_fit = df_for_fit[X + [y]].astype(float)

    # to make coefficients comparable
    df_for_fit[X] = df_for_fit[X].apply(lambda x: (x - x.mean()) / x.std())

    # Prepare the independent variables
    X_multi = df_for_fit[X]
    X_multi_const = sm.add_constant(X_multi)
    y_predict = df_for_fit[y]
    return X_multi_const, y_predict


def logi_model_fit(df: pd.DataFrame, X, y, method='newton'):
    X_multi_const, y_predict = logi_model_fit_input(df, X, y, method=method)

    # Fit the logistic regression model with multiple regressors
    logit_model_multi = sm.Logit(y_predict, X_multi_const).fit(method=method)

    # Display the summary, which includes p-values for all regressors
    results = logit_model_multi.summary(xname= ["intercept"] + X)
    
    return results, logit_model_multi


def hierarchical_partitioning(df, x_cols, y_col, method='newton'):
    contributions = defaultdict(list)
    n_vars = len(x_cols)

    for k in range(1, n_vars + 1):
        for subset in itertools.combinations(x_cols, k): # generate all the combinations of the variables
            subset = list(subset)
            # fit model input and generate model 
            X, y = logi_model_fit_input(df, subset, y_col)
            model = sm.Logit(y, X).fit(method=method)
            
            r2 = r2_score(y, model.predict(X))

            # contribution for each variable in the combinations
            for var in subset:
                reduced_subset = subset.copy()
                reduced_subset.remove(var)
                X_reduced = X[reduced_subset] if reduced_subset else X['const'] # if no feature after remove only calcalate constant
                model_reduced = sm.Logit(y, X_reduced).fit(method=method)
                r2_reduced = r2_score(y, model_reduced.predict(X_reduced))
                delta = r2 - r2_reduced
                contributions[var].append(delta)
    
    # normalize the contributions
    avg_contrib = {var: np.mean(contrib) for var, contrib in contributions.items()}
    total = sum(avg_contrib.values())
    norm_contrib = {var: val / total for var, val in avg_contrib.items()}
    return pd.Series(norm_contrib)


def drop_one_var_contribution(df, x_cols, y_col, method='newton'):
    """
    Calculate the contribution of each variable in a logistic regression model
    by dropping one variable at a time and measuring the change in R^2.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_cols (list): List of independent variable column names.
        y_col (str): Dependent variable column name.
        method (str): Optimization method for fitting the model.
    Returns:
        pd.Series: Normalized contributions of each variable.
    """
    # Calculate r^2 for the full model
    contributions = {var: [] for var in x_cols}
    X, y = logi_model_fit_input(df, x_cols, y_col)
    model = sm.Logit(y, X).fit(method=method)
    r2 = r2_score(y, model.predict(X))
    # Iterate over each variable, drop it, and calculate the change in r^2
    for var in x_cols:
        X_reduced = X.drop(columns=[var])
        model_reduced = sm.Logit(y, X_reduced).fit(method=method)
        r2_reduced = r2_score(y, model_reduced.predict(X_reduced))
        delta = r2 - r2_reduced
        contributions[var].append(delta)
    # normalize the contributions
    avg_contrib = {var: np.mean(contrib) for var, contrib in contributions.items()}
    total = sum(avg_contrib.values())
    norm_contrib = {var: val / total for var, val in avg_contrib.items()}
    return pd.Series(norm_contrib)


def verify_params_time_kernel(dic:dict, y:str, max_lag_range=range(1,10), tau_range=range(1,10)):
    combinations = list(itertools.product(max_lag_range, tau_range))
    comb_dict = {}
    # iterate all the combinations of max_lag and tau of time kernel
    for comb in combinations:
        max_lag = comb[0]
        tau = comb[1]
        previous_impact_on_kernel_mice = []
        for df_name, df in zip(dic.keys(), dic.values()):
            if y == 'first_choice_numeric':
                # df = add_mouse_first_choice(df)
                df['first_choice_numeric'] = df['first_choice'].apply(
                        transform_side_choice_to_numeric
                        )
            elif y == 'correct_numeric':
                df['correct_numeric'] = df['correct'].astype(int)
            else:
                raise ValueError("impact should be either 'first_choice_numeric' or 'correct_numeric'")
            
            for session in df['session'].unique():
                df_session = df[df['session'] == session]
                df_session['previous_impact_on_kernel'] = previous_impact_on_time_kernel(df_session[y], max_lag=max_lag, tau=tau)
                df.loc[df_session.index, 'previous_impact_on_kernel'] = df_session['previous_impact_on_kernel']
            _, model = logi_model_fit(df, X=['previous_impact_on_kernel'], y=y)
            previous_impact_on_kernel_mice.append(abs(model.params['previous_impact_on_kernel']))
        comb_dict[comb] = np.mean(previous_impact_on_kernel_mice)
    return comb_dict


def filter_variables_for_model(dic_fit:dict, X:list, y:str, max_lag=None, tau=None):
    """
    This function filters the variables for the logistic regression model.
    Calculate the correlation matrix and the contribution of each variable
    by dropping one variable at a time. For different subjects, calculate the mean values.
    Args:
        dic_fit (dict): Dictionary with dataframes for each subject.
        X (list): List of independent variable column names.
        y (str): Dependent variable column name.
        max_lag (int, optional): Maximum lag for the time kernel impact.
        tau (int, optional): Time constant for the time kernel impact.
    Returns:
        corr_mat_list (list): List of correlation matrices for each subject.
        norm_contribution_df (pd.DataFrame): DataFrame with normalized contributions of each subject
    """
    corr_mat_list = []
    norm_contribution_df = pd.DataFrame([])
    for df_name, df_for_fit in zip(dic_fit.keys(), dic_fit.values()):
        # if there is "time_kernel_impact" in the variables, get the time kernel impact
        if (max_lag is not None) & (tau is not None):
            df_for_fit = get_time_kernel_impact(df_for_fit, y=y, max_lag=max_lag, tau=tau)
        # calculate the correlation matrix of each two variables
        corr_fit_X_df = df_for_fit[X].corr()
        corr_mat_list.append(corr_fit_X_df)
        # calculate the contribution of each variable by dropping one variable at a time
        norm_contribution = drop_one_var_contribution(df_for_fit, x_cols = X, y_col = y, method='bfgs')
        norm_contribution_df[df_name] = norm_contribution

    return corr_mat_list, norm_contribution_df
