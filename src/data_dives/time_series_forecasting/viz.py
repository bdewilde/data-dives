from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm


def plot_time_series(
    mseries: Sequence[pd.Series],
    labels: Optional[Sequence[str]] = None,
    subplots: bool = True,
    plot_kw: Optional[Dict[str, Any]] = None,
    **fig_kw,
) -> plt.Subplot | List[plt.Subplot]:
    """
    Make time plots for multiple time series, either on separate subplots or overlaid.

    Args:
        mseries: Sequence of one or more time series.
        labels: Identifying labels for each series in ``mseries``, used as y-axis labels
            if ``subplots`` is True or legend labels otherwise.
        subplots: If True, plot each series separately on subplots; otherwise, plot
            them all together on a single axis.
        plot_kw: Keyword arguments passed on to :func:`matplotlib.axes.Axes.plot()`.
        **fig_kw: Keyword arguments passed on to :func:`plt.figure()`.

    Returns:
        Subplot (or a list thereof) on which series have been plotted.
    """
    labels = itertools.repeat(None, len(mseries)) if labels is None else labels
    plot_kw = {} if plot_kw is None else plot_kw
    if subplots is True:
        _, axes = plt.subplots(nrows=len(mseries), sharex=True, **fig_kw)
        for ax, series, label in zip(axes, mseries, labels):
            _ = series.plot.line(ax=ax, ylabel=label, **plot_kw)
            # ax.xaxis_date()
        return list(axes)
    else:
        _, ax = plt.subplots(**fig_kw)
        for series, label in zip(mseries, labels):
            _ = series.plot.line(ax=ax, label=label, **plot_kw)
        # ax.xaxis_date()
        _ = ax.legend(loc="best")
        return ax


def plot_seasonal_periods(
    data_grped: pd.core.groupby.SeriesGroupBy,
    dt_attr: str,  # Literal["hour", day", "month", "quarter", "year"],
    *,
    cmap: str = "viridis",
    plot_kw: Optional[Dict[str, Any]] = None,
    **fig_kw,
) -> plt.Subplot:
    """
    Make a plot of each seasonal period in ``data_grped`` laid one on top of the other,
    with optional colormap (+colorbar).

    Args:
        data_grped: Grouped time series, as produced by :meth:`pd.Series.groupby()`.
        dt_attr: Datetime component accessible as group index attribute.
        cmap: Name of cmap to use in coloring each seasonal periods' line.
        plot_kw: Keyword arguments passed on to :func:`matplotlib.axes.Axes.plot()`.
        **fig_kw: Keyword arguments passed on to :func:`plt.figure()`.

    Returns:
        Subplot on which seasonal periods have been plotted.
    """
    fig, ax = plt.subplots(**fig_kw)
    plot_kw = {} if plot_kw is None else plot_kw
    cmap = plt.cm.get_cmap(name=cmap)
    group_names = data_grped.groups.keys()
    cmap_norm = mpl.colors.Normalize(vmin=min(group_names), vmax=max(group_names))
    _ = fig.colorbar(plt.cm.ScalarMappable(norm=cmap_norm, cmap=cmap), ax=ax)
    ngroups = data_grped.ngroups
    minx = None
    maxx = None
    for i, (name, grp) in enumerate(data_grped):
        xvals = getattr(grp.index, dt_attr)
        yvals = grp.values.astype(float)
        color = cmap(i / ngroups) if cmap else None
        _ = ax.plot(xvals, yvals, color=color, label=name, **plot_kw)
        minx = min(minx, min(xvals)) if minx is not None else min(xvals)
        maxx = max(maxx, max(xvals)) if maxx is not None else max(xvals)
        ylabel = grp.name
    _ = ax.set_xlim(left=minx, right=maxx)
    _ = ax.set_xlabel(dt_attr)
    _ = ax.set_ylabel(ylabel or "")
    return ax


def plot_autocorrelations(
    series: pd.Series,
    *,
    lags: Optional[int] = None,
    alpha: float = 0.05,
    use_vlines: bool = True,
    zero: bool = False,
    acf_kw: Optional[Dict[str, Any]] = None,
    pacf_kw: Optional[Dict[str, Any]] = None,
    **fig_kw,
) -> List[plt.Subplot]:
    """
    Plot full and partial autocorrelations for a time series in a single figure
    with shared lags on the x-axis.

    Args:
        series: Time series data whose autocorrelations are to be plotted.
        lags
        alpha
        use_vlines
        acf_kw
        pacf_kw
        **fig_kw: Keyword arguments passed on to :func:`plt.figure()`.

    Returns:
        Subplots on which autocorrelations have been plotted, as a list.

    See Also:
        - :func:`sm.graphics.tsa.plot_acf()`
        - :func:`sm.graphics.tsa.plot_pacf()`
    """
    acf_kw = {} if acf_kw is None else acf_kw
    pacf_kw = {} if pacf_kw is None else pacf_kw
    if "title" not in acf_kw:
        acf_kw["title"] = None
    if "title" not in pacf_kw:
        pacf_kw["title"] = None
    _, axes = plt.subplots(nrows=2, sharex=True, **fig_kw)
    _ = sm.graphics.tsa.plot_acf(
        series,
        lags=lags, alpha=alpha, use_vlines=use_vlines, zero=zero, ax=axes[0],
        **acf_kw
    )
    _ = sm.graphics.tsa.plot_pacf(
        series,
        lags=lags, alpha=alpha, use_vlines=use_vlines, zero=zero, ax=axes[1],
        **pacf_kw
    )
    axes[0].set_ylabel("acf")
    axes[1].set_ylabel("pacf")
    axes[1].set_xlabel("lag")
    return list(axes)


def plot_residuals_diagnostics(
    residuals: pd.Series,
    acf_kw=None,
    hist_kw=None,
    **fig_kw
) -> List[plt.Subplot]:
    """
    Make three diagnostic plots of residuals to evaluate a model's fit: residuals as
    a time series, an autocorrelation correlogram, and a histogram as compared to a
    normal distribution.

    Args:
        residuals
        acf_kw
        hist_kw
        **fig_kw

    Returns:
        Subplots on which residuals diagnostics have been plotted.
    """
    fig = plt.figure(**fig_kw)
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
