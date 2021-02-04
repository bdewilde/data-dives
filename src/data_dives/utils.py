from __future__ import annotations

import importlib.resources
import pathlib
import urllib.parse
from typing import Optional

_PKG_NAME = "data_dives"


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
    """
    Get full path to package's default data directory on disk.

    Note:
        Automatically makes directory if it doesn't already exist.
    """
    with importlib.resources.path(_PKG_NAME, "__init__.py") as fpath:
        dirpath = fpath.parent.parent.parent.joinpath("data")
    dirpath.mkdir(exist_ok=True)
    return dirpath
