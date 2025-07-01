import numpy as np
import pandas as pd
from scipy.stats import norm


def basel_backtesting(
    losses: pd.DataFrame, var_dict: dict[int:float] | dict[tuple:float], confidence: float = 0.99
) -> pd.DataFrame:
    """
    Perform Basel backtesting on a DataFrame of losses against a dictionary of VaR thresholds.
    This function counts the number of violations for each stock and determines the zone
    based on the number of violations relative to the expected number of violations.

    Parameters
    ----------
    losses : pd.DataFrame
        A pandas DataFrame containing loss values for different stocks,
        indexed by stock identifiers.
    var_dict : dict[int:float] | dict[tuple:float]
        A dictionary where keys are stock identifiers (or tuples of identifiers)
        and values are the corresponding Value at Risk (VaR) thresholds.
    confidence : float, optional
        The confidence level for the Value at Risk calculation, default is 0.99.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the count of violations and the corresponding zone for each stock.
        The zones are categorized as 'green', 'yellow', or 'red' based on the number of violations.
    """
    violation_counts = pd.Series(
        {col: (losses[col].dropna() > var_dict[col]).sum() for col in losses.columns}
    )
    var_col = pd.Series({col: var_dict[col] for col in losses.columns if col in var_dict})

    def determine_zone(
        violation_count: int,
        num_observations: int,
        threshold: float = 0.99,
    ) -> str:
        """
        Determine the zone based on the violation count and number of observations.
        """
        mean = num_observations * (1 - threshold)
        std_dev = np.sqrt(num_observations * threshold * (1 - threshold))

        green_zone_num = int(norm.ppf(0.95) * std_dev + mean)
        red_zone_num = int(norm.ppf(0.99) * std_dev + mean)

        if violation_count <= green_zone_num:
            return "green"
        elif violation_count <= red_zone_num:
            return "yellow"
        else:
            return "red"

    zones = pd.Series(
        {
            col: determine_zone(violation_count, losses[col].dropna().shape[0], confidence)
            for col, violation_count in violation_counts.items()
        }
    )

    violation_df = pd.DataFrame(
        {"VaR": var_col, "violation_count": violation_counts, "zone": zones}
    )
    violation_df["zone"] = violation_df["zone"].astype("category")
    violation_df["zone"] = violation_df["zone"].cat.set_categories(
        ["green", "yellow", "red"], ordered=True
    )

    return violation_df.sort_values(by="zone")


def sener_backtesting(
    losses: pd.DataFrame, var_dict: dict[int:float] | dict[tuple:float], theta: float
) -> list[float]:
    """
    Compute the Sener backtesting penalty for a DataFrame of
    losses against a dictionary of VaR thresholds.

    Parameters
    ----------
    losses : pd.DataFrame
        A pandas DataFrame containing loss values for different stocks,
        indexed by stock identifiers.
    var_dict : dict[int:float] | dict[tuple:float]
        A dictionary where keys are stock identifiers (or tuples of identifiers)
        and values are the corresponding Value at Risk (VaR) thresholds.
    theta : float
        A weighting factor between 0 and 1 that determines the balance
        between violation space measure and safe space measure.

    Returns
    -------
    tuple[list[float], list[float]]
        A tuple containing two lists:
        - Individual penalties for each stock.
        - Aggregate penalty across all stocks.
        Each penalty is a tuple of (violation space measure, safe space measure).
    """

    def compute_violation_penalty(losses: pd.Series, var: float) -> float:
        # Step 1: compute violations
        losses_reset = losses.reset_index(drop=True)
        epsilon = losses_reset - var
        epsilon[epsilon <= 0] = np.nan

        # Step 2: find clusters of consecutive violations
        violations = epsilon.dropna()
        violation_indices = violations.index

        clusters = []
        current_cluster = []

        for _, idx in enumerate(violation_indices):
            if not current_cluster:
                current_cluster.append(idx)
            elif idx == current_cluster[-1] + 1:
                current_cluster.append(idx)
            else:
                clusters.append(current_cluster)
                current_cluster = [idx]
        if current_cluster:
            clusters.append(current_cluster)

        # Step 3: compute C_r for each cluster
        cluster_list = []
        for cluster in clusters:
            eps = epsilon.loc[cluster].values
            c_r = np.prod(1 + eps) - 1
            cluster_list.append((cluster[0], c_r, eps))  # store first index for timing

        # Step 4: compute Phi
        phi = 0
        h = len(cluster_list)
        for r in range(h - 1):
            t_r, c_r, eps_r = cluster_list[r]
            for s in range(1, h - r):
                t_rs, _, eps_rs = cluster_list[r + s]
                k = t_rs - t_r
                prod = (np.prod(1 + eps_r) * np.prod(1 + eps_rs)) - 1
                phi += prod / k

        return phi

    def compute_safe_penalty(losses: pd.Series, var: float) -> float:
        return np.sum(var - losses[(losses < var) & (losses > 0)])

    indiv_penalty_list = []
    agg_penalty_list = []

    for col in losses.columns:
        var_col = var_dict.get(col, float("inf"))
        indiv_penalty_list.append(
            (
                compute_violation_penalty(losses[col], var_col),
                compute_safe_penalty(losses[col], var_col),
            )
        )
        agg_penalty_list.append(
            theta * compute_violation_penalty(losses[col], var_col)
            + (1 - theta) * compute_safe_penalty(losses[col], var_col)
        )

    var_col = pd.Series({col: var_dict[col] for col in losses.columns if col in var_dict})
    violation_df = pd.DataFrame(
        {
            "VaR": var_col,
            "violation_penalty": [vio for vio, _ in indiv_penalty_list],
            "safe_penalty": [safe for _, safe in indiv_penalty_list],
            "agg_penalty": agg_penalty_list,
        }
    )

    return violation_df
