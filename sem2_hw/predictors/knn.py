from typing import Union
import numpy as np


class KNN:
    _win_size: int
    _neigbours: int
    _metric: str
    _points: Union[np.ndarray, None]
    _labels: Union[np.ndarray, None]

    def __init__(self, win_size: int = 4, neigbours: int = 4, metric: str = 'decart') -> None:
        if metric != 'decart' and metric != 'Manhattan':
            raise TypeError("wrong metric")
        self._metric = metric
        if not isinstance(neigbours, int):
            raise TypeError("not integer")
        if neigbours <= 0 or neigbours // 1 != neigbours:
            print(
                f"your neughbours are uncorrect(<0 or not an integer)."
                f" We cast it into{abs(neigbours)//1 + 1}"
            )
            self._neigbours = abs(neigbours) // 1 + 1
        else:
            self._neigbours = neigbours
        if not isinstance(win_size, int):
            raise TypeError("not integer")
        if win_size <= 0 or win_size // 1 != win_size:
            print(
                f"your win_size is uncorrect(<0 or not integer)."
                f" We cast it into{abs(win_size)//1 + 1}"
            )
            self._win_size = abs(win_size) // 1 + 1

        else:
            self._win_size = win_size

    def fit(self, points: np.ndarray, labels: np.ndarray) -> None:
        if len(points) != len(labels):
            raise RuntimeError("different len")
        if isinstance(points, np.ndarray) and isinstance(labels, np.ndarray):
            self._points = points
            self._labels = labels

        else:
            raise TypeError("it is not a np.ndarray")

    def predict(
            self,
            points: np.ndarray,
    ):
        print(points.shape,'---',self._points.shape)#-----
        if not isinstance(points, np.ndarray):
            raise TypeError("not np.array")
        if len(points.shape) == 1:  # перевод одномерного случая в двумерный массив координат
            points = points.reshape(points.shape[0], 1)
            self._points = self._points.reshape(self._points.shape[0], 1)
        if self._metric == 'decart':
            distances = np.linalg.norm(
                self._points - points[:, np.newaxis], axis=2)
        if self._metric == 'Manhattan':
            distances = np.linalg.norm(
                self._points - points[:, np.newaxis], axis=2, ord=1)
        print(distances.shape)#-------
        k_windows = np.sort(distances).T[self._win_size]

        kerE = np.where(np.abs(np.sort(distances).T / k_windows).T <= 1,
                        0.75 * (1 - (np.sort(distances).T / k_windows).T**2), 0)
        print(kerE.shape)

        mask = np.argsort(distances)
        presort = self._labels[mask]
        presort = presort[::, 0:self._neigbours]
        kerE = kerE[::, 0:self._neigbours]
        print(presort.shape, kerE.shape)
        ans = np.zeros((presort.shape[0], presort.shape[0]))
        for i in np.unique(self._labels):

            sums = np.sum(np.where(presort == i, 1, 0) * kerE, axis=1)
            a = np.where(sums > ans[1], 1, 0)

            b = np.where(sums < ans[1], 1, 0)

            ans[0] = ans[0] * b + (np.zeros(presort.shape[0]) + i) * a

            ans[1] = ans[1] * b + sums * a

        return ans[0]
