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
