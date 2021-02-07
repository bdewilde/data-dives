from __future__ import annotations

import pathlib
from typing import Optional

import pandas as pd

from data_dives.datasets import base
from data_dives import utils


_DATASET_INFO = {
    "name": "GISTEMP",
    "meta": {
        "site_url": "https://data.giss.nasa.gov/gistemp",
        "description": (
            "Estimates of global surface temperature change based on combined land-surface air and "
            "sea-surface water temperature anomalies, expressed as a Land-Ocean Temperature Index (LOTI) "
            "measured relative to average temps over 1951-1980 for the given place and time of year. "
            "Global mean with monthly resolution, 1880 – Present."
        ),
        "citation": (
            "Lenssen, N., G. Schmidt, J. Hansen, M. Menne, A. Persin, R. Ruedy, and D. Zyss, 2019: "
            "Improvements in the GISTEMP uncertainty model. J. Geophys. Res. Atmos., 124, no. 12, 6307-6326, "
            "doi:10.1029/2018JD029522."
        )
    },
    "download_url": "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
}


class GISTEMP(base.Dataset):

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
        Load GISTEMP dataset data, either from disk or a URL,
        and automatically save it to disk if loaded from URL.
        """
        fpath = utils.get_fpath(data_dir, fname=None, url=self.download_url)
        data = base.load_csv_data(
            fpath=fpath,
            url=self.download_url,
            force=force,
            read_from_url_kwargs={"skiprows": 1},
        )
        if not fpath.exists() or force is True:
            data.to_csv(fpath, header=True, index=False)
        return data

    def prepare(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Args:
            data: As loaded from :meth:`GISTEMP.load()`.

        Returns:
            GISTEMP dataset data, prepared for analysis.
        """
        data = (
            data
            .drop(columns=["J-D", "D-N", "DJF", "MAM", "JJA", "SON"])
            .set_index("Year")
            .stack(level=0)
            .to_frame(name="LOTI")
        )
        data = (
            data
            .set_index(
                pd.to_datetime(data.index.map(lambda x: f"{x[1]} {x[0]}"), format="%b %Y")
            )
            .resample("MS")
            .mean()
            .rename_axis("dt")
        )
        return data
