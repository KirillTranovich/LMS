import numpy as np
from typing import Any, Callable

def get_boxplot_outliers(
    data: np.ndarray,
    key: Callable[[Any], Any],
) -> np.ndarray:
    if not isinstance(data, np.ndarray):
        raise TypeError("there is not np.ndarray")

    if len(data.shape) == 1:  # нам в любом случае нужен двумерный массив координат
        data = data.reshape(data.shape[0], 1)
    sort_data = np.sort(data.copy(), axis=0)#жестко пережделать
    qwart1 = np.array(sort_data[int(sort_data.shape[0] * 0.25)][0])
    qwart3 = np.array(sort_data[int(sort_data.shape[0] * 0.75)][0])
    tunnel = (qwart3 - qwart1) * 1.5
    mask1 = np.where(data >= qwart1 - tunnel, 1, 0)#1- a-ok 0-bad
    mask1 = np.where(np.sum(mask1, axis=1) // 1 == 0, 0, 1)
    mask2 = np.where(data <= qwart3 + tunnel, 1, 0)
    mask2 = np.where(np.sum(mask2, axis=1) // 1 == 0, 0, 1)
    ans_mask = np.where(mask1 + mask2 == 2, 1, 0)
    return np.where(ans_mask == 1, 0, 1)
