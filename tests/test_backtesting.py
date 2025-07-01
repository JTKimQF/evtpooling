import pandas as pd
import pytest
from scipy.stats import uniform

from evtpooling import basel_backtesting

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
