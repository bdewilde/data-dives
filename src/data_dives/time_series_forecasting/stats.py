from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import sklearn.base
import sklearn.metrics
import statsmodels.api as sm


def trend_strength(trend: pd.Series, residual: pd.Series) -> float:
    """
    Compute the strength of a decomposed time series' trend component by comparing
    it to the residual component.

    Args:
        trend
        residual

    Returns:
        Trend strength, where 0.0 is weakest and 1.0 is strongest.
    """
    return max(1 - residual.var() / (residual.var() + trend.var()), 0.0)


def seasonal_strength(seasonal: pd.Seris, residual: pd.Series) -> float:
    """
    Compute the strength of a decomposed time series' seasonal component by comparing
    it to the residual component.

    Args:
        seasonal
        resid

    Returns:
        Seasonal strength, where 0.0 is weakest and 1.0 is strongest.
    """
    return max(1 - residual.var() / (residual.var() + seasonal.var()), 0.0)


def adjusted_r2_score(
    y: np.array, yhat: np.array, model: sklearn.base.BaseEstimator
) -> float:
    """
    Compute the adjusted R^2 metric (aka coefficient of determination), which measures
    the "goodness of fit" of a linear model as the proportion of variation in the
    dependent variable accounted for by the independent variable(s).

    It's less biased than (unadjusted R^2), and better for use in evaluating models.

    Args:
        y: True dependent variable values
        yhat: Estimated dependent variable values
        model: Fit model

    Returns:
        Adjusted R^2 score.

    See Also:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
    """
    r2 = sklearn.metrics.r2_score(y, yhat)
    n = len(y)
    p = len(model.coef_)
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))


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
