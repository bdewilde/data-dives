from __future__ import annotations

import importlib.resources
import pathlib
import urllib.parse
from typing import Optional

_PKG_NAME = "data_dives"


def get_fpath(
    data_dir: Optional[str | pathlib.Path],
    fname: Optional[str],
    url: Optional[str],
) -> pathlib.Path:
    data_dir = get_default_data_dir() if data_dir is None else to_path(data_dir)
    if fname:
        return data_dir.resolve().joinpath(fname)
    elif url:
        fname = get_fname_from_url(url)
        return data_dir.resolve().joinpath(fname)
    else:
        raise ValueError("either `fname` or `url` must be specified")


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
