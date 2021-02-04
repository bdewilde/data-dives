import itertools

import pandas as pd


def naive_forecast(series: pd.Series, steps: int) -> pd.Series:
    future_index = _make_future_index(series.index, steps)
    last_val = series.iat[-1]
    values = [last_val] * steps
    return pd.Series(data=values, index=future_index, name=series.name)


def naive_seasonal_forecast(series: pd.Series, period: int, steps: int) -> pd.Series:
    future_index = _make_future_index(series.index, steps)
    values = list(itertools.islice(itertools.cycle(series.tail(period).tolist()), steps))
    return pd.Series(data=values, index=future_index, name=series.name)


def drift_forecast(series: pd.Series, steps: int) -> pd.Series:
    # manual implementation of y = m*x + b
    b = series.iat[0]
    dy = series.iat[-1] - series.iat[0]
    dx = (series.index[-1] - series.index[0]).days
    m = dy / dx
    future_index = _make_future_index(series.index, steps)
    x = (future_index - series.index[0]).days
    values = m * x + b
    return pd.Series(data=values, index=future_index, name=series.name)


def _make_future_index(index: pd.DatetimeIndex, steps: int) -> pd.DatetimeIndex:
    max_dt = index.max()
    return pd.date_range(start=max_dt + index.freq, freq=index.freq, periods=steps)


# def linear_forecast(series: pd.Series, steps: int) -> pd.Series:
#     poly = (
#         np.polynomial.Polynomial([1, 1])
#         .fit(
#             (series.index.astype("int64") // (10**9)).to_numpy(float),
#             series.to_numpy(float),
#             1
#         )
#     )
#     last_dt = series.index.max()
#     index = pd.date_range(start=last_dt + series.index.freq, freq=series.index.freq, periods=steps)
#     values = [poly(idx) for idx in (index.astype("int64") // (10**9)).to_numpy(int)]
#     return pd.Series(data=values, index=index, name=series.name)
