import numpy as np
from typing import Any, Callable


def get_boxplot_outliers(
    data: np.ndarray,
    key: Callable[[Any], Any],
) -> np.ndarray:
    if not isinstance(data, np.ndarray):
        raise TypeError("there is not np.ndarray")
    sort_data = key(data.copy())
    qwart1 = np.array(sort_data[int(sort_data.size * 0.25)])
    qwart3 = np.array(sort_data[int(sort_data.size * 0.75)])
    tunnel = (qwart3 - qwart1) * 1.5
    mask1 = np.where(data >= qwart1 - tunnel, 1, 0)
    mask1 = np.where(np.sum(mask1, axis=1) != mask1.shape[1], 0, 1)
    mask2 = np.where(data <= qwart3 + tunnel, 1, 0)
    mask2 = np.where(np.sum(mask2, axis=1) != mask1.shape[1], 0, 1)
    ans_mask = np.where(mask1 + mask2 == 2, 1, 0)
    return np.argwhere(ans_mask == 1)[:, 0]
