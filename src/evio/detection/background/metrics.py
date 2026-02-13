import numpy as np


def update_stats_ema(
    values: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    count: np.ndarray,
    alpha: float,
) -> None:
    if not np.any(mask):
        return
    delta = values[mask] - mean[mask]
    mean[mask] += alpha * delta
    var[mask] = (1.0 - alpha) * var[mask] + alpha * (delta * delta)
    count[mask] += 1


def update_stats_welford(
    values: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    m2: np.ndarray,
    count: np.ndarray,
) -> None:
    if not np.any(mask):
        return
    count[mask] += 1
    delta = values[mask] - mean[mask]
    mean[mask] += delta / count[mask]
    delta2 = values[mask] - mean[mask]
    m2[mask] += delta * delta2
    var_vals = np.zeros_like(mean[mask])
    np.divide(
        m2[mask],
        count[mask] - 1,
        out=var_vals,
        where=count[mask] > 1,
    )
    var[mask] = var_vals


def compute_coactivity(
    microstate: np.ndarray,
    *,
    sub_rows: int,
    sub_cols: int,
    time_bins: int,
) -> np.ndarray:
    cells = microstate.shape[0]
    active = (
        microstate.reshape(cells, sub_rows, sub_cols, time_bins).sum(axis=3)
        > 0
    )
    active_i = active.astype(np.int8, copy=False)
    padded = np.pad(active_i, ((0, 0), (1, 1), (1, 1)), mode="constant")
    neighbors = (
        padded[:, :-2, :-2]
        + padded[:, :-2, 1:-1]
        + padded[:, :-2, 2:]
        + padded[:, 1:-1, :-2]
        + padded[:, 1:-1, 2:]
        + padded[:, 2:, :-2]
        + padded[:, 2:, 1:-1]
        + padded[:, 2:, 2:]
    )
    return (neighbors * active_i).sum(axis=(1, 2)).astype(np.float32)
