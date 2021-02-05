from __future__ import annotations

from typing import Optional

import pandas as pd
import statsmodels.api as sm


def adfuller_test(
    series: pd.Series,
    *,
    regression: str = "c",
    maxlag: Optional[int] = None,
    autolag: Optional[str] = "AIC",
) -> pd.Series:
    """
    Run an Augmented Dickey-Fuller unit root test on ``series`` to test for stationarity.

    See Also:
        :func:`sm.tsa.stattools.adfuller()`
    """
    result = sm.tsa.stattools.adfuller(
        series, regression=regression, maxlag=maxlag, autolag=autolag, store=False
    )
    test_stat, pval, num_lags, num_obs, crit_vals, _ = result
    index = (
        ["stationarity", "test statistic", "p-value", "num lags", "num obs"] +
        [f"critical value {key}" for key in crit_vals.keys()]
    )
    data = (
        [pval < 0.05, test_stat, pval, num_lags, num_obs] +
        [val for val in crit_vals.values()]
    )
    return pd.Series(index=index, data=data)


def kpss_test(
    series: pd.Series,
    *,
    regression: str = "c",
    nlags: Optional[int | str] = "auto",
) -> pd.Series:
    """
    Run a Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test on ``series`` for stationarity.

    See Also:
        :func:`sm.tsa.stattools.adfuller()`
    """
    result = sm.tsa.stattools.kpss(
        series, regression=regression, nlags=nlags, store=False
    )
    test_stat, pval, num_lags, crit_vals = result
    index = (
        ["stationarity", "test statistic", "p-value", "num lags"] +
        [f"critical value {key}" for key in crit_vals.keys()]
    )
    data = (
        [pval > 0.05, test_stat, pval, num_lags] +
        [val for val in crit_vals.values()]
    )
    return pd.Series(index=index, data=data)
