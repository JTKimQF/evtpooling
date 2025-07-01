import numpy as np
import pandas as pd
import pytest
from scipy.stats import uniform

from evtpooling import basel_backtesting, sener_backtesting

# --------------------
# Fixtures (mock data)
# --------------------


@pytest.fixture
def basel_backtesting_data():
    losses_col = [
        (1, "EUR", "01W"),
        (2, "EUR", "01W"),
        (3, "EUR", "01W"),
        (4, "EUR", "01W"),
        (5, "EUR", "01W"),
    ]

    var_dict = {col: var for col, var in zip(losses_col, [10, 10, 10, 10, 10], strict=False)}

    df_losses = pd.DataFrame(
        {
            losses_col[0]: [uniform.rvs(0, 9.99) for _ in range(200)],
            losses_col[1]: [uniform.rvs(0, 9.99) for _ in range(197)] + [11.0, 12.0, 13.0],
            losses_col[2]: [uniform.rvs(0, 9.99) for _ in range(196)] + [10.1, 120.0, 20.0, 50.0],
            losses_col[3]: [uniform.rvs(0, 9.99) for _ in range(195)]
            + [uniform.rvs(10.1, 20.0) for _ in range(5)],
            losses_col[4]: [uniform.rvs(0, 9.99) for _ in range(190)]
            + [uniform.rvs(10.1, 20.0) for _ in range(10)],
        }
    )

    return df_losses, var_dict


@pytest.fixture
def sener_backtesting_data():
    losses_col = [
        (1, "EUR", "01W"),
        (2, "EUR", "01W"),
        (3, "EUR", "01W"),
    ]

    var_dict = {col: var for col, var in zip(losses_col, [10, 10, 10], strict=False)}

    df_losses = pd.DataFrame(
        {
            losses_col[0]: [11, 5, 12, 12, 10],
            losses_col[1]: [6, 7, 8, 9, 10],
            losses_col[2]: [11, 11, 12, 10, 15],
        }
    )

    return df_losses, var_dict


# --------------------
# Tests
# --------------------


def test_basel_backtesting(basel_backtesting_data):
    df_losses, var_dict = basel_backtesting_data
    result = basel_backtesting(df_losses, var_dict, confidence=0.99)

    # Check the result DataFrame structure
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(var_dict)
    assert "violation_count" in result.columns
    assert "zone" in result.columns

    # Check the zones based on expected violations
    for _, row in result.iterrows():
        if row["violation_count"] <= 4:
            assert row["zone"] == "green"
        elif row["violation_count"] <= 5:
            assert row["zone"] == "yellow"
        else:
            assert row["zone"] == "red"


def test_sener_backtesting(sener_backtesting_data):
    df_losses, var_dict = sener_backtesting_data
    vio_df = sener_backtesting(df_losses, var_dict, theta=0.5)

    assert vio_df.shape[0] == len(var_dict)
    assert np.array_equal(vio_df["violation_penalty"].values, [8.5, 0, 17.75])
    assert np.array_equal(vio_df["safe_penalty"].values, [5, 10, 0])
    assert np.array_equal(vio_df["agg_penalty"].values, [6.75, 5, 8.875])
