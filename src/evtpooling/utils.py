import numpy as np
import pandas as pd


def get_alpha_var(
    losses: pd.Series, k_values: np.ndarray | int, threshold: float = 0.99
) -> tuple[list[float], list[float]]:
    """
    Calculate the Hill estimator and Value at Risk (VaR) for given losses.

    Parameters:
    ----------
    losses : pd.Series
        A pandas Series containing the loss values.
    k_values : list
        A list of integers representing the range of threshold to compute the Hill estimator.
    threshold : float
        The threshold for the Value at Risk calculation, typically between 0 and 1.

    Returns:
    --------
    tuple
        A tuple containing two lists:
        - alpha_hat_list: The Hill estimator values for each k in k_values.
        - var_hat_list: The corresponding Value at Risk estimates.
    """
    # Initialize the list to store the sorted values
    var_hat_list = []
    alpha_hat_list = []

    if isinstance(k_values, int):
        k_values = [k_values]

    for k in k_values:
        top_k_logs = np.log(losses.iloc[-k:])  # Log of the largest k values
        log_x_nk = np.log(losses.iloc[-k - 1])  # Log of the (k+1)-th largest value
        alpha_hat = 1 / (np.mean(top_k_logs) - log_x_nk)  # Hill estimator
        alpha_hat_list.append(alpha_hat)
        # VaR estimator formula
        var = losses.iloc[-k] * (k / (len(losses) * (1 - threshold))) ** (1 / alpha_hat)
        var_hat_list.append(var)

    return alpha_hat_list, var_hat_list
