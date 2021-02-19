import math

import numpy as np
import pandas as pd


def get_bootstrapped_samples(
    series: pd.Series,
    block_size: int,
    strategy: str = "mb",  # Literal["mb", "cb"]
    n_samples: int = 1,
) -> pd.DataFrame:
    """
    Get one or multiple bootstrapped samples as concatenated blocks of contiguous values
    of ``series`` randomly partitioned using a block bootstrapping strategy.

    Args:
        series
        block_size
        strategy
        n_samples

    Returns:
        DataFrame of bootstrapped samples of shape (len(series), n_samples).
    """
    strategy_funcs = {"mb": get_mb_indexes, "cb": get_cb_indexes}  # clunky, ugh
    strategy_func = strategy_funcs[strategy]
    n_obs = len(series)
    values = series.to_numpy()
    bootstrap_data = np.empty((n_obs, n_samples))
    for i in range(n_samples):
        sample_idxs = strategy_func(n_obs, block_size)
        sample_values = values[sample_idxs]
        bootstrap_data[:, i] = sample_values
    return pd.DataFrame(
        data=bootstrap_data,
        index=series.index,
        columns=[f"sample{i}" for i in range(n_samples)],
    )


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
