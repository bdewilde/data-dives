from __future__ import annotations

import pathlib
from typing import Optional

import pandas as pd

from data_dives.datasets import base
from data_dives import utils


_DATASET_INFO = {
    "name": "Beijing PM2.5",
    "meta": {
        "site_url": "https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data",
        "description": (
            "Particulate matter of size 2.5µm or less (PM2.5) measurements made by the US Embassy "
            "in Beijing, as well as meteorological data from Beijing Capital International Airport. "
            "Hourly resolution, from 2010-01-01 to 2014-12-31."
        ),
        "citation": (
            "Liang, X., Zou, T., Guo, B., Li, S., Zhang, H., Zhang, S., Huang, H. and Chen, S. X. (2015). "
            "Assessing Beijing's PM2.5 pollution: severity, weather impact, APEC and winter heating. "
            "Proceedings of the Royal Society A, 471, 20150257."
        )
    },
    "download_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv",
}


class BeijingPM25(base.Dataset):

    def __init__(self):
        super().__init__(
            _DATASET_INFO["name"],
            _DATASET_INFO["meta"],
            _DATASET_INFO["download_url"],
        )

    def load(
        self,
        data_dir: Optional[str | pathlib.Path] = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """
        Load Beijing PM2.5 dataset data, either from disk or a URL,
        and automatically save it to disk if loaded from URL.
        """
        fpath = utils.get_fpath(data_dir, fname=None, url=self.download_url)
        data = base.load_csv_data(
            fpath=fpath,
            url=self.download_url,
            force=force,
        )
        if not fpath.exists() or force is True:
            data.to_csv(fpath, header=True, index=False)
        return data

    def prepare(
        self,
        data: pd.DataFrame,
        *,
        freq: str = "1H",
        fill: str = "interpolate",  # Literal["forward", "interpolate"]
    ) -> pd.DataFrame:
        """
        Args:
            data: As loaded from :meth:`BeijingPM25.load()`.
            freq: Time series frequency as a ``pandas``-style "offset alias" string,
                to which data will be resampled using mean() to aggregate values.
            fill: Name of method with which to fill missing values.
                "forward" => :meth:`pd.DataFrame.fillna(method="ffill")`
                "interpolate" => :meth:`pd.DataFrame.interpolate(method="time")`

        Returns:
            BeijingPM25 dataset data, prepared for analysis.

        See Also:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        """
        data = (
            data
            # build combined datetime column
            .assign(
                dt=lambda df: pd.to_datetime(
                    df[["year", "month", "day", "hour"]], format="%Y %m %d %H"
                )
            )
            # drop unnecessary cols and set dt as index
            .drop(columns=["No", "year", "month", "day", "hour"])
            .set_index("dt")
            # rename columns, for clarity
            .rename(
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
        )
        # resample and aggregate values, with special handling for categorical wind_dir
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
            .astype({"wind_dir": "string"})
        )
        # fill missing values using specified method
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
