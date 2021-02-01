from __future__ import annotations

import importlib.resources
import pathlib
import urllib.parse
from typing import Optional


def to_path(str_or_path: str | pathlib.Path) -> pathlib.Path:
    """If possible / as needed, convert ``str_or_path`` into a :class:`pathlib.Path`."""
    if isinstance(str_or_path, str):
        return pathlib.Path(str_or_path)
    elif isinstance(str_or_path, pathlib.Path):
        return str_or_path
    else:
        raise TypeError()


def get_fname_from_url(url: str) -> Optional[str]:
    """
    Extract a filename from a URL's path, if possible.

    Warning:
        This function doesn't have guardrails, so only use if you know you have a valid
        filename at the end of the URL path.
    """
    return pathlib.Path(urllib.parse.urlparse(url).path).name


def get_default_data_dir() -> pathlib.Path:
    # dirpath = pathlib.Path(__file__).parent.parent.parent.resolve().joinpath("data")
    pkgname = "time_series_forecasting_study"
    with importlib.resources.path(pkgname, "__init__.py") as fpath:
        dirpath = fpath.parent.parent.parent.joinpath("data")
    dirpath.mkdir(exist_ok=True)
    return dirpath
