from __future__ import annotations

import pathlib
from typing import Any, Dict, Optional

import pandas as pd


class Dataset:
    """
    Base class for datasets.

    Args:
        name
        meta
        download_url

    Attributes:
        name
        meta
        download_url
    """

    def __init__(self, name: str, meta: Dict[str, Any], download_url: str):
        self.name = name
        self.meta = meta
        self.download_url = download_url

    def __repr__(self):
        return f"Dataset('{self.name}')"

    @property
    def info(self) -> Dict[str, Any]:
        """Name, metadata, and download URL for dataset."""
        return {
            "name": self.name,
            **self.meta,
            "download_url": self.download_url,
        }

    def load(
        self,
        data_dir: Optional[str | pathlib.Path] = None,
        force: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load dataset data, either from disk or a URL, and automatically save it to disk
        if loaded from URL.
        """
        raise NotImplementedError

    def prepare(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Args:
            data: As loaded in :meth:`Dataset.load()`.
            **kwargs

        Returns:
            Dataset data, prepared for analysis.
        """
        raise NotImplementedError


def load_csv_data(
    *,
    fpath: pathlib.Path,
    url: str,
    force: bool = False,
    read_from_url_kwargs: Optional[Dict[str, Any]] = None,
    read_from_disk_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    if not fpath.exists() or force is True:
        data = pd.read_csv(url, **(read_from_url_kwargs or {}))
    else:
        data = pd.read_csv(fpath, **(read_from_disk_kwargs or {}))
    return data
