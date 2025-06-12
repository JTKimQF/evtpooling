import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evtpooling.utils import get_alpha_var


def alpha_var_sing_plot(
    losses: pd.Series,
    k_values: list[int],
    threshold: float = 0.99,
    name: str = "",
    file_path: str = None,
) -> None:
    """
    Plot the Hill estimator and Value at Risk (VaR) for a single series of losses in one plot.

    Parameters:
    ----------
    losses : pd.Series
        A pandas Series containing the loss values.
    k_values : list[int]
        A list of integers representing the range of threshold to compute the Hill estimator.
    threshold : float, optional
        The threshold for the Value at Risk calculation, typically between 0 and 1. Default is 0.99.
    name : str, optional
        A name for the plot, used in the title. Default is an empty string.
    file_path : str, optional
        The file path where the plot will be saved. If empty, the plot will not be saved.

    Returns:
    -------
    None
        Displays the plot with the Hill estimator and VaR.
    """
    stocks = losses.dropna().sort_values()
    alpha_hat_list, var_hat_list = get_alpha_var(stocks, k_values, threshold)
    file_path = file_path.rstrip("/\\")

    _, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Plotting the VaR plot
    sns.lineplot(x=k_values, y=var_hat_list, ax=ax, color="b", label="VaR (in %)")
    ax.set_title(f"VaR and Alpha plot {name}")
    ax.set_xlabel("Threshold Index")
    ax.set_ylabel("VaR (in %)")
    ax.tick_params(axis="y", color="b")
    ax.set_xlim(k_values[0] - 10, k_values[-1] + 10)
    ax.legend_.remove()
    ax.xaxis.grid(True)

    # Plotting the Hill estimator
    ax2 = ax.twinx()
    sns.lineplot(x=k_values, y=alpha_hat_list, ax=ax2, color="r", label="Alpha")
    ax2.set_ylabel("Alpha")
    ax2.tick_params(axis="y", color="r")
    ax2.legend_.remove()

    # 🔁 Combine both legends
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    if file_path:
        if name:
            plt.savefig(os.path.join(file_path, f"alpha_var_plot_{name}.png"))
        else:
            plt.savefig(os.path.join(file_path, "alpha_var_plot.png"))

    plt.show()


def alpha_var_agg_plot(
    losses: pd.Series,
    k_values: list[int],
    threshold: float = 0.99,
    vline: int = None,
    file_path: str = "",
    details_on: bool = False,
    dict_gvkey_conm: dict[int, str] = None,
) -> None:
    """
    Plot the Hill estimator and Value at Risk for multiple series of losses in two separate plots.

    Parameters:
    ----------
    losses : pd.DataFrame
        A pandas DataFrame containing multiple series of loss values, indexed by stock identifiers.
    k_values : list[int]
        A list of integers representing the range of threshold to compute the Hill estimator.
    threshold : float, optional
        The threshold for the Value at Risk calculation, typically between 0 and 1. Default is 0.99.
    vline : int, optional
        A vertical line to be drawn at a specific k value in the plots. Default is None.
    file_path : str, optional
        The file path where the plots will be saved. If empty, plots will not be saved.
        Default is an empty string.
    details_on : bool, optional
        If True, additional details will be shown in the plots, such as titles and legends.
        Default is False.
    dict_gvkey_conm : dict[int, str], optional
        A dictionary mapping stock identifiers (gvkeys) to company names (conms).
        If provided, company names will be used in the plots. Default is None.

    Returns:
    -------
    None
        Displays two plots: one for VaR and one for the Hill estimator (alpha).
    """
    # --- Prep Lists ---
    alpha_hat_list = []
    var_hat_list = []
    gvkey_list = []
    k_values_list = np.tile(k_values, losses.shape[1])
    file_path = file_path.rstrip("/\\")

    # --- Estimate alpha and VaR for each stock ---
    for stocks in losses.columns:
        stocks_clean = losses[stocks].dropna().sort_values()
        alpha_hat, var_hat = get_alpha_var(stocks_clean, k_values, threshold)
        alpha_hat_list.extend(alpha_hat)
        var_hat_list.extend(var_hat)
        gvkey = stocks[0] if isinstance(stocks, tuple) else stocks
        gvkey_list.extend([gvkey] * len(k_values))

    # --- Construct DataFrame ---
    df_alpha_var = pd.DataFrame(
        {
            "gvkey": gvkey_list,
            "k_values": k_values_list,
            "alpha_hat": alpha_hat_list,
            "var_hat": var_hat_list,
        }
    )

    # --- Add company names if provided ---
    plot_kwargs = {"data": df_alpha_var, "x": "k_values"}

    if dict_gvkey_conm is not None:
        conmlist = [dict_gvkey_conm[gvkey] for gvkey in gvkey_list]
        df_alpha_var["conm"] = conmlist
        plot_kwargs["hue"] = "conm"
        plot_kwargs["style"] = "conm"
        plot_kwargs["palette"] = "tab10"

    sns.set_style("whitegrid")

    # === VaR Plot ===
    _, ax = plt.subplots(figsize=(15, 9))
    sns.lineplot(y="var_hat", ax=ax, **plot_kwargs)
    ax.set_xlabel("k", fontsize=28)
    ax.set_ylabel("VaR (%)", fontsize=28)
    ax.set_xlim(k_values[0] - 10, k_values[-1] + 10)
    ax.tick_params(axis="y", labelsize=24)
    ax.tick_params(axis="x", labelsize=24)

    if details_on:
        ax.set_title("VaR Plot")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        if ax.legend_:
            ax.legend_.remove()

    if vline is not None:
        ax.axvline(vline, 0, color="r", linestyle="--")

    if file_path:
        plt.savefig(os.path.join(file_path, "var_plot.png"))

    plt.show()

    # === Alpha Plot ===
    _, ax2 = plt.subplots(figsize=(15, 9))
    sns.lineplot(y="alpha_hat", ax=ax2, **plot_kwargs)
    ax2.set_xlabel("k", fontsize=28)
    ax2.set_ylabel("Alpha", fontsize=28)
    ax2.set_xlim(k_values[0] - 10, k_values[-1] + 10)
    ax2.tick_params(axis="y", labelsize=24)
    ax2.tick_params(axis="x", labelsize=24)

    if details_on:
        ax2.set_title("Alpha Plot")
        box2 = ax2.get_position()
        ax2.set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])
        ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        if ax2.legend_:
            ax2.legend_.remove()

    if vline is not None:
        ax2.axvline(vline, 0, color="r", linestyle="--")

    if file_path:
        plt.savefig(os.path.join(file_path, "alpha_plot.png"))

    plt.show()
