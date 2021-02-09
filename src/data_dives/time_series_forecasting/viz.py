from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm


def plot_time_series(
    mseries: Sequence[pd.Series],
    labels: Optional[Sequence[str]] = None,
    subplots: bool = True,
    plot_kwargs: Optional[Dict[str, Any]] = None,
    **fig_kwargs,
) -> plt.Subplot | List[plt.Subplot]:
    """
    Make time plots for multiple time series, either on separate subplots or overlaid.

    Args:
        mseries: Sequence of one or more time series.
        labels: Identifying labels for each series in ``mseries``, used as y-axis labels
            if ``subplots`` is True or legend labels otherwise.
        subplots: If True, plot each series separately on subplots; otherwise, plot
            them all together on a single axis.
        plot_kwargs: Keyword arguments passed on to :func:`matplotlib.axes.Axes.plot()`.
        **fig_kwargs: Keyword arguments passed on to :func:`plt.figure()`.

    Returns:
        Subplot (or a list thereof) on which series have been plotted.
    """
    labels = itertools.repeat(None, len(mseries)) if labels is None else labels
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    if subplots is True:
        _, axes = plt.subplots(nrows=len(mseries), sharex=True, **fig_kwargs)
        for ax, series, label in zip(axes, mseries, labels):
            _ = series.plot.line(ax=ax, ylabel=label, **plot_kwargs)
            # ax.xaxis_date()
        return list(axes)
    else:
        _, ax = plt.subplots(**fig_kwargs)
        for series, label in zip(mseries, labels):
            _ = series.plot.line(ax=ax, label=label, **plot_kwargs)
        # ax.xaxis_date()
        _ = ax.legend(loc="best")
        return ax


def plot_autocorrelations(
    series: pd.Series,
    *,
    lags: Optional[int] = None,
    alpha: float = 0.05,
    use_vlines: bool = True,
    zero: bool = False,
    acf_kwargs: Optional[Dict[str, Any]] = None,
    pacf_kwargs: Optional[Dict[str, Any]] = None,
    **fig_kwargs,
) -> List[plt.Subplot]:
    """
    Plot full and partial autocorrelations for a time series in a single figure
    with shared lags on the x-axis.

    Args:
        series: Time series data whose autocorrelations are to be plotted.
        lags
        alpha
        use_vlines
        acf_kwargs
        pacf_kwargs
        **fig_kwargs: Keyword arguments passed on to :func:`plt.figure()`.

    Returns:
        Subplots on which autocorrelations have been plotted, as a list.

    See Also:
        - :func:`sm.graphics.tsa.plot_acf()`
        - :func:`sm.graphics.tsa.plot_pacf()`
    """
    acf_kwargs = {} if acf_kwargs is None else acf_kwargs
    pacf_kwargs = {} if pacf_kwargs is None else pacf_kwargs
    if "title" not in acf_kwargs:
        acf_kwargs["title"] = None
    if "title" not in pacf_kwargs:
        pacf_kwargs["title"] = None
    _, axes = plt.subplots(nrows=2, sharex=True, **fig_kwargs)
    _ = sm.graphics.tsa.plot_acf(
        series,
        lags=lags, alpha=alpha, use_vlines=use_vlines, zero=zero, ax=axes[0],
        **acf_kwargs
    )
    _ = sm.graphics.tsa.plot_pacf(
        series,
        lags=lags, alpha=alpha, use_vlines=use_vlines, zero=zero, ax=axes[1],
        **pacf_kwargs
    )
    axes[0].set_ylabel("acf")
    axes[1].set_ylabel("pacf")
    axes[1].set_xlabel("lag")
    return list(axes)


def plot_residuals_diagnostics(
    residuals: pd.Series,
    acf_kw=None,
    hist_kw=None,
    **fig_kwargs
) -> List[plt.Subplot]:
    """
    Make three diagnostic plots of residuals to evaluate a model's fit: residuals as
    a time series, an autocorrelation correlogram, and a histogram as compared to a
    normal distribution.

    Args:
        residuals
        acf_kw
        hist_kw
        **fig_kwargs

    Returns:
        Subplots on which residuals diagnostics have been plotted.
    """
    fig = plt.figure(**fig_kwargs)
    gs = fig.add_gridspec(nrows=2, ncols=2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    # time plot up top
    _ = residuals.plot.line(ylabel="residuals", ax=ax1)
    # autocorrelation correlogram bottom left
    acf_kw = acf_kw or {}
    acf_title = acf_kw.pop("title", None)
    _ = sm.graphics.tsa.plot_acf(residuals, ax=ax2, title=acf_title, **acf_kw)
    if not acf_title:
        ax2.set_ylabel("acf")
    # histogram density + normal distribution bottom right
    hist_kw = hist_kw or {}
    _ = ax3.hist(residuals, density=True, label="residuals", **hist_kw)
    mu = 0
    sigma = residuals.std()
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    ax3.plot(x, st.norm.pdf(x, mu, sigma), label="N(0, $\sigma$)")
    ax3.legend()
    return [ax1, ax2, ax3]
