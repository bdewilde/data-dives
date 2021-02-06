from __future__ import annotations

import pathlib
from typing import Optional

import pandas as pd

from data_dives.datasets import base
from data_dives import utils


_DATASET_INFO = {
    "name": "MLO CO2",
    "meta": {
        "site_url": "https://scrippsco2.ucsd.edu/data/atmospheric_co2/mlo.html",
        "description": (
            "In-situ CO2 measurements taken at Mauna Loa Observatory, Hawaii "
            "(Latitude 19.5°N, Longitude 155.6°W, Elevation 3397m). "
            "Daily resolution, from 1958 – Present."
        ),
        "citation": (
            "C. D. Keeling, S. C. Piper, R. B. Bacastow, M. Wahlen, T. P. Whorf, M. Heimann, and H. A. Meijer, "
            "Exchanges of atmospheric CO2 and 13CO2 with the terrestrial biosphere and oceans from 1978 to 2000. "
            "I. Global aspects, SIO Reference Series, No. 01-06, Scripps Institution of Oceanography, San Diego, "
            "88 pages, 2001."
        )
    },
    "download_url": "https://scrippsco2.ucsd.edu/assets/data/atmospheric/stations/in_situ_co2/daily/daily_in_situ_co2_mlo.csv",
}


class MLOCO2(base.Dataset):

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
        Load MLO CO2 dataset data, either from disk or a URL,
        and automatically save it to disk if loaded from URL.
        """
        fpath = utils.get_fpath(data_dir, fname=None, url=self.download_url)
        data = base.load_csv_data(
            fpath=fpath,
            url=self.download_url,
            force=force,
            read_from_url_kwargs={"skiprows": 33},
        )
        if not fpath.exists() or force is True:
            data.to_csv(fpath, header=True, index=False)
        return data

    def prepare(
        self,
        data: pd.DataFrame,
        *,
        freq: str = "1D",
        fill: str = "interpolate",  # Literal["forward", "interpolate"]
    ) -> pd.DataFrame:
        """
        Args:
            data: As loaded from :meth:`MLOCO2.load()`.
            freq: Time series frequency as a ``pandas``-style "offset alias" string,
                to which data will be resampled using mean() to aggregate values.
            fill: Name of method with which to fill missing values.
                "forward" => :meth:`pd.DataFrame.fillna(method="ffill")`
                "interpolate" => :meth:`pd.DataFrame.interpolate(method="time")`

        Returns:
            MLO CO2 dataset data, prepared for analysis.

        See Also:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        """
        data = (
            data
            # some of the columns have extra chars, so strip em
            .rename(columns=lambda x: x.strip("% "))
            # build combined datetime column, fix the CO2 column so it reads as numeric
            .assign(
                dt=lambda df: pd.to_datetime(
                    df[["Yr", "Mn", "Dy"]].rename(
                        columns={"Yr": "year", "Mn": "month", "Dy": "day"}
                    )
                ),
                CO2=lambda df: pd.to_numeric(df["CO2"].str.strip().replace("NaN", pd.NA))
            )
            # drop unnecessary cols and set dt as index
            .drop(columns=["Yr", "Mn", "Dy", "NB", "scale"])
            .set_index("dt")
        )
        # resample and aggregate values
        data = data.resample(freq, axis="index").mean()
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
        data = data.dropna(axis="index", how="any", subset=["CO2"])
        return data
