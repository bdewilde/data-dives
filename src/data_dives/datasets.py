from __future__ import annotations

import pathlib
from typing import Optional

import pandas as pd

from . import utils


URL_MLO_CO2 = "https://scrippsco2.ucsd.edu/assets/data/atmospheric/stations/in_situ_co2/daily/daily_in_situ_co2_mlo.csv"  # noqa
URL_BEIJING_PM25 = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"  # noqa


def _load_csv_dataset(
    *,
    data_dir: Optional[str | pathlib.Path],
    fname: str,
    url: str,
    force: bool = False,
    **read_csv_from_url_kwargs,
) -> pd.DataFrame:
    if data_dir is None:
        data_dir = utils.get_default_data_dir()
    fpath = utils.to_path(data_dir).joinpath(fname)
    if not fpath.exists() or force is True:
        data = pd.read_csv(url, **read_csv_from_url_kwargs)
        data.to_csv(fpath, header=True, index=False)
    else:
        data = pd.read_csv(fpath)
    return data


def load_mlo_co2(
    data_dir: Optional[str | pathlib.Path] = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Args:
        data_dir
        force

    Returns:
        "Raw" Mauna Loa Observatory in-situ CO2 dataset.

    Reference:
        https://scrippsco2.ucsd.edu/data/atmospheric_co2/mlo.html
    """
    return _load_csv_dataset(
        data_dir=data_dir,
        fname=utils.get_fname_from_url(URL_MLO_CO2),
        url=URL_MLO_CO2,
        force=force,
        skiprows=33,
    )


def munge_mlo_co2(
    data: pd.DataFrame,
    *,
    freq: str = "1D",
    fill: str = "interpolate",  # Literal["forward", "interpolate"]
) -> pd.DataFrame:
    """
    Args:
        data: As loaded from :func:`load_mlo_co2()`.
        freq: Time series frequency as a ``pandas``-style "offset alias" string,
            to which data will be resampled using mean() to aggregate values.
        fill: Name of method with which to fill missing values.
            "forward" => :meth:`pd.DataFrame.fillna(method="ffill")`
            "interpolate" => :meth:`pd.DataFrame.interpolate(method="time")`

    Returns:
        Munged Mauna Loa Observatory in-situ CO2 dataset.

    See Also:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
    """
    # some of the columns have extra chars, so strip em
    data = data.rename(columns=lambda x: x.strip("% "))
    # build a combined datetime column, and fix the CO2 column so it reads as numeric
    data = data.assign(
        dt=pd.to_datetime(
            data[["Yr", "Mn", "Dy"]].rename(
                columns={"Yr": "year", "Mn": "month", "Dy": "day"}
            )
        ),
        CO2=pd.to_numeric(data["CO2"].str.strip().replace("NaN", pd.NA))
    )
    # drop unnecessary cols and set dt as index
    data = data.drop(columns=["Yr", "Mn", "Dy", "NB", "scale"])
    data = data.set_index("dt")
    # resample and fill missing values
    data = data.resample(freq, axis="index").mean()
    if fill == "forward":
        data = data.fillna(method="ffill", limit=None)
    elif fill == "interpolate":
        data = data.interpolate(method="time", limit=None)
    else:
        raise ValueError()
    # convert dtypes here at the end since .interpolate(method="time") doesn't work
    # for pandas' fancy new dtypes Int64 and Float64 (at least as of v1.1)
    data = data.convert_dtypes()
    # drop any rows from start/end for which key value wasn't filled
    data = data.dropna(axis="index", how="any", subset=["CO2"])
    return data


def load_beijing_pm25(
    data_dir: Optional[str | pathlib.Path] = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Args:
        data_dir
        force

    Returns:
        "Raw" Beijing PM2.5 dataset.

    Reference:
        https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
    """
    return _load_csv_dataset(
        data_dir=data_dir,
        fname=utils.get_fname_from_url(URL_BEIJING_PM25),
        url=URL_BEIJING_PM25,
        force=force,
    )


def munge_beijing_pm25(
    data: pd.DataFrame,
    *,
    freq: str = "1D",
    fill: str = "interpolate",  # Literal["forward", "interpolate"]
) -> pd.DataFrame:
    """
    Args:
        data: As loaded from :func:`load_beijing_pm25()`.
        freq: Time series frequency as a ``pandas``-style "offset alias" string,
            to which data will be resampled using mean() to aggregate values.
        fill: Name of method with which to fill missing values.
            "forward" => :meth:`pd.DataFrame.fillna(method="ffill")`
            "interpolate" => :meth:`pd.DataFrame.interpolate(method="time")`

    Returns:
        Munged Beijing PM2.5 dataset.

    See Also:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
    """
    # build a combined datetime column
    data = data.assign(
        dt=pd.to_datetime(data[["year", "month", "day", "hour"]], format="%Y %m %d %H")
    )
    # drop unnecessary cols and set dt as index
    data = data.drop(columns=["No", "year", "month", "day", "hour"])
    data = data.set_index("dt")
    # rename columns, for clarity
    data = data.rename(
        columns={
            "DEWP": "dew_point",
            "TEMP": "temp",
            "PRES": "pressure",
            "cbwd": "wind_dir",
            "Iws": "wind_speed",
            "Is": "hrs_snow",
            "Ir": "hrs_rain",
        }
    )
    # resample and fill missing values, with special handling for categorical wind_dir
    data = (
        data
        .resample(freq, axis="index")
        .agg(
            {
                "pm2.5": "mean",
                "dew_point": "mean",
                "temp": "mean",
                "pressure": "mean",
                "wind_dir": lambda x: x.mode(),
                "wind_speed": "mean",
                "hrs_snow": "mean",  # "max"
                "hrs_rain": "mean",  # "max"
            }
        )
    )
    data = data.astype({"wind_dir": "category"})
    if fill == "forward":
        data = data.fillna(method="ffill", limit=None)
    elif fill == "interpolate":
        data = data.interpolate(method="time", limit=None)
    else:
        raise ValueError()
    # convert dtypes here at the end since .interpolate(method="time") doesn't work
    # for pandas' fancy new dtypes Int64 and Float64 (at least as of v1.1)
    data = data.convert_dtypes()
    # drop any rows from start/end for which key value wasn't filled
    data = data.dropna(axis="index", how="any", subset=["pm2.5"])
    return data
