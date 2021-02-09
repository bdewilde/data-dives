import statsmodels.tsa.seasonal


def trend_strength(decomp: statsmodels.tsa.seasonal.DecomposeResult) -> float:
    """
    Compute the strength of a time series trend component from its fit decomposition.

    Args:
        decomp

    Returns:
        Trend strength, where 0.0 is weakest and 1.0 is strongest.
    """
    return max(1 - decomp.resid.var() / (decomp.resid.var() + decomp.trend.var()), 0.0)


def seasonal_strength(decomp: statsmodels.tsa.seasonal.DecomposeResult) -> float:
    """
    Compute the strength of a time series seasonal component from its fit decomposition.

    Args:
        decomp

    Returns:
        Seasonal strength, where 0.0 is weakest and 1.0 is strongest.
    """
    return max(1 - decomp.resid.var() / (decomp.resid.var() + decomp.seasonal.var()), 0.0)
