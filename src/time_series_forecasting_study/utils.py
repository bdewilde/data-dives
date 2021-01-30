from __future__ import annotations

import pathlib


def to_path(str_or_path: str | pathlib.Path) -> pathlib.Path:
    """If possible / as needed, convert ``str_or_path`` into a :class:`pathlib.Path`."""
    if isinstance(str_or_path, str):
        return pathlib.Path(str_or_path)
    elif isinstance(str_or_path, pathlib.Path):
        return str_or_path
    else:
        raise TypeError()
