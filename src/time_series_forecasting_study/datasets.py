from __future__ import annotations

import pathlib

import pandas as pd

from . import utils


_URL_BEIJING_PM25 = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"  # noqa


def load_beijing_pm25(
    data_dir: str | pathlib.Path,
    force: bool = False,
) -> pd.DataFrame:
    """
    Args:
        data_dir
        force

    Returns:
        "Raw" Beijing PM2.5 dataset.
    """
    fpath = utils.to_path(data_dir).joinpath("PRSA_data_2010.1.1-2014.12.31.csv")
    if not fpath.exists() or force is True:
        data = pd.read_csv(_URL_BEIJING_PM25)
        data.to_csv(fpath, header=True, index=False)
    else:
        data = pd.read_csv(fpath)
    data = data.convert_dtypes()
    return data


def munge_beijing_pm25(
    data: pd.DataFrame,
    freq: str = "1D",
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """
    Args:
        data: As loaded from :func:`load_beijing_pm25()`.

    Returns:
        Munged Beijing PM2.5 dataset.
    """
    data.loc[:, "dt"] = pd.to_datetime(
        data[["year", "month", "day", "hour"]],
        format="%Y %m %d %H",
    )
    data = data.drop(columns=["No", "year", "month", "day", "hour"])
    data = data.set_index("dt")
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
    # we have to handle wind_dir specially, so this won't cut it
    # data = data.resample("1D", axis="index").mean()
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
    # pandas makes us re-cast this column to string for some reason
    data = data.astype({"wind_dir": "string"})
    if fill_method == "ffill":
        data = data.fillna(method="ffill", limit=None)
    elif fill_method == "interpolate":
        data = data.interpolate(method="time", limit=None)
    else:
        raise ValueError()
    # data = data.iloc[1:, :]
    data = data.dropna(axis="index", how="any", subset=["pm2.5"])
    return data
