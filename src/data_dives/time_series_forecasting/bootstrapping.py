from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd


def bootstrap(
    *arrays,
    block_size: int,
    strategy: str = "mb",  # Literal["mb", "cb"]
) -> pd.Series | pd.DataFrame | List[pd.Series | pd.DataFrame]:
    """
    Get bootstrapped samples of each array in ``arrays`` as concatenated blocks of
    contiguous values, randomly partitioned via a block bootstrapping strategy.

    Args:
        *arrays: Sequence of series or dataframes with the same number of rows.
        block_size: Size of each block to concatenate.
        strategy: Name of blocked boostrapping strategy; either "mb" for "moving block"
            or "cb" for "circular block".

    Returns:
        Bootstrapped arrays in input order, each of the same shape and type
        as the original data; if only a single array is passed in, the output is also
        a single bootstrapped array rather than a list of them.
    """
    strategy_funcs = {"mb": get_mb_indexes, "cb": get_cb_indexes}  # clunky, ugh
    strategy_func = strategy_funcs[strategy]
    if len({len(array) for array in arrays}) > 1:
        raise ValueError("all arrays to be resampled must have the same length")
    n_obs = len(arrays[0])
    sample_idxs = strategy_func(n_obs, block_size)
    bootstrapped_arrays = []
    for array in arrays:
        if isinstance(array, pd.Series):
            bootstrapped_arrays.append(
                pd.Series(
                    data=array.iloc[sample_idxs].to_numpy(),
                    index=array.index,
                    name=array.name,
                )
            )
        elif isinstance(array, pd.DataFrame):
            bootstrapped_arrays.append(
                pd.DataFrame(
                    data=array.iloc[sample_idxs].to_numpy(),
                    index=array.index,
                    columns=array.columns,
                )
            )
        else:
            raise TypeError()
    if len(bootstrapped_arrays) == 1:
        bootstrapped_arrays = bootstrapped_arrays[0]
    return bootstrapped_arrays


def get_mb_indexes(n_obs: int, block_size: int) -> np.ndarray:
    """
    Get bootstrapped indexes using the moving block bootstrap strategy,
    given the total number of observations in a series and desired block size.

    Args:
        n_obs
        block_size

    Returns:
        Array of bootstrapped indexes of shape (n_obs,).
    """
    n_blocks = math.ceil(n_obs / block_size)
    max_block_idx = n_obs - block_size
    # randomly sample from all indexes to get block starting points
    block_start_idxs = np.random.randint(
        0, high=max_block_idx, size=(n_blocks, 1), dtype=int
    )
    # generate one sequence per block to be added to start indexes
    block_next_idxs = np.repeat([np.arange(0, block_size)], n_blocks, axis=0)
    # compute block indexes
    block_idxs = block_start_idxs + block_next_idxs
    # flatten indexes into single array and chop off any remainder
    return np.ravel(block_idxs)[:n_obs]


def get_cb_indexes(n_obs: int, block_size: int) -> np.ndarray:
    """
    Get bootstrapped indexes using the circular block bootstrap strategy,
    given the total number of observations in a series and desired block size.

    Args:
        n_obs
        block_size

    Returns:
        Array of bootstrapped indexes of shape (n_obs,).
    """
    n_blocks = math.ceil(n_obs / block_size)
    max_block_idx = n_obs
    # randomly sample from all indexes to get block starting points
    block_start_idxs = np.random.randint(
        0, high=max_block_idx, size=(n_blocks, 1), dtype=int
    )
    # generate one sequence per block to be added to start indexes
    block_next_idxs = np.repeat([np.arange(0, block_size)], n_blocks, axis=0)
    # compute block indexes, looping around over crossed sequence boundaries
    block_idxs = np.mod(block_start_idxs + block_next_idxs, max_block_idx)
    # flatten indexes into single array and chop off any remainder
    return np.ravel(block_idxs)[:n_obs]
