from __future__ import annotations

import datetime
from typing import Hashable, Optional, Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.deterministic import DeterministicTerm


class PiecewiseLinearTrend(DeterministicTerm):

    def __init__(self, knots: Sequence[str] | Sequence[int] | Sequence[datetime.datetime]):
        self.knots = tuple(pd.to_datetime(knots))

    def __str__(self):
        return f"PiecewiseLinearTrend(knots=self.knots)"

    def _eq_attr(self):
        return (self.knots,)

    def in_sample(self, index: pd.Index) -> pd.DataFrame:
        """
        Produce deterministic trends for in-sample fitting.

        Args:
            index: Pandas datetime index, or convertable into one.

        Returns:
            DataFrame containing deterministic piecewise linear terms.
        """
        index = _to_datetime_index(index)
        nobs = index.shape[0]
        terms = np.zeros((nobs, len(self.knots) + 1))
        terms[:, 0] = np.arange(start=1, stop=nobs + 1, step=1)
        for i, knot in enumerate(self.knots):
            start_idx = index.get_slice_bound(knot, side="left", kind="getitem")
            terms[:, i + 1] = np.clip(terms[:, 0] - start_idx, 0, None)
        return pd.DataFrame(
            data=terms,
            index=index,
            columns=self._get_column_names(),
        )

    def out_of_sample(
        self,
        steps: int,
        index: pd.Index,
        forecast_index: Optional[pd.Index] = None,
    ) -> pd.DataFrame:
        """
        Produce deterministic trends for out-of-sample forecasts.

        Args:
            steps: Number of steps to forecast.
            index: Pandas datetime index, or convertable into one.
            forecast_index: Pandas datetime index, or convertable into one.
                If provided, must have ``steps`` elements.

        Returns:
            DataFrame containing deterministic piecewise linear terms.
        """
        fcast_index = self._extend_index(index, steps, forecast_index)
        nobs = index.shape[0]
        terms = np.zeros((steps, len(self.knots) + 1))
        terms[:, 0] = np.arange(start=nobs + 1, stop=nobs + steps + 1, step=1)
        for i, knot in enumerate(self.knots):
            start_idx = index.get_slice_bound(knot, side="left", kind="getitem")
            terms[:, i + 1] = np.clip(terms[:, 0] - start_idx, 0, None)
        return pd.DataFrame(
            data=terms,
            index=fcast_index,
            columns=self._get_column_names(),
        )

    def _get_column_names(self):
        return [f"trend({i})" for i in range(len(self.knots) + 1)]


class DatetimeAttributeSeasonality(DeterministicTerm):

    _is_dummy = True

    def __init__(self, dt_attr: str):
        self.dt_attr = dt_attr

    def __str__(self):
        return f"DatetimeAttributeSeasonality('{self.dt_attr}')"

    def _eq_attr(self):
        return (self.dt_attr,)

    def in_sample(self, index: pd.Index) -> pd.DataFrame:
        """
        Produce deterministic trends for in-sample fitting.

        Args:
            index: Pandas datetime index, or convertable into one.

        Returns:
            DataFrame containing deterministic piecewise linear terms.
        """
        index = _to_datetime_index(index)
        dt_values = getattr(index, self.dt_attr)
        data = pd.get_dummies(
            dt_values,
            prefix=f"{self.dt_attr}=",
            prefix_sep="",
            # NOTE: first dummy variable is dropped by statsmodels when _is_dummy=True
            # so we don't do it here in pandas
            drop_first=False,
        )
        return pd.DataFrame(data=data).set_index(index)

    def out_of_sample(
        self,
        steps: int,
        index: pd.Index,
        forecast_index: Optional[pd.Index] = None,
    ) -> pd.DataFrame:
        """
        Produce deterministic trends for out-of-sample forecasts.

        Args:
            steps: Number of steps to forecast.
            index: Pandas datetime index, or convertable into one.
            forecast_index: Pandas datetime index, or convertable into one.
                If provided, must have ``steps`` elements.

        Returns:
            DataFrame containing deterministic piecewise linear terms.
        """
        fcast_index = self._extend_index(index, steps, forecast_index)
        dt_values = getattr(fcast_index, self.dt_attr)
        data = pd.get_dummies(
            dt_values, prefix=f"{self.dt_attr}=", prefix_sep="", drop_first=True
        )
        return pd.DataFrame(data=data).set_index(fcast_index)


def _to_datetime_index(index: pd.Index | Sequence[Hashable]) -> pd.DatetimeIndex:
    """Coerce ``index`` into a :class:`pd.DatetimeIndex`, or crash trying."""
    if isinstance(index, pd.DatetimeIndex):
        return index
    else:
        return pd.DatetimeIndex(pd.to_datetime(index))
