import numpy as np
import matplotlib.pyplot as plt
from typing import Any
import os


def _create_diagrams(
    axis: plt.Axes,
    abs: np.ndarray,
    diagram_type: Any,
    orient="vertical",
) -> None:
    if diagram_type == "box":
        axis.boxplot(
            abs,
            vert=orient != "vertical",
            patch_artist=True,
            boxprops=dict(facecolor="lightsteelblue"),
            medianprops=dict(color="k"),
        )
    elif diagram_type == "hist":
        axis.hist(
            abs,
            bins=50,
            color="cornflowerblue",
            orientation=orient,
            density=True,
            alpha=0.5,
        )
    elif diagram_type == "violin":
        violin_parts = axis.violinplot(
            abs,
            vert=orient != "vertical",
            showmedians=True,
        )
        for body in violin_parts["bodies"]:
            body.set_facecolor("cornflowerblue")
            body.set_edgecolor("blue")

        for part in violin_parts:
            if part == "bodies":
                continue

            violin_parts[part].set_edgecolor("cornflowerblue")
    else:
        raise TypeError("Wrong Type")


def visualize_distribution(
    data: np.ndarray,
    diagram_type: Any,
    path_to_save: str = "",
) -> None:
    if not isinstance(data, np.ndarray):
        raise TypeError("there is not np.ndarray")

    if len(data.shape) == 1:
        figure, axis = plt.subplots(figsize=(8, 8))
        _create_diagrams(axis, data, diagram_type)
        plt.show()

    if len(data.shape) == 2 and data.shape[1] == 2:
        abscissa = data[:, 0]
        ordinates = data[:, 1]
        figure = plt.figure(figsize=(8, 8))
        grid = plt.GridSpec(4, 4, wspace=0.2, hspace=0.2)

        axis_scatter = figure.add_subplot(grid[:-1, 1:])
        axis_vert = figure.add_subplot(
            grid[:-1, 0],
            sharey=axis_scatter,
        )
        axis_hor = figure.add_subplot(
            grid[-1, 1:],
            sharex=axis_scatter,
        )
        axis_scatter.scatter(abscissa, ordinates,
                             color="cornflowerblue", alpha=0.5)
        _create_diagrams(axis_hor, abscissa, diagram_type)
        _create_diagrams(axis_vert, ordinates,
                         diagram_type, orient="horizontal",)

        axis_hor.invert_yaxis()
        axis_vert.invert_xaxis()
        plt.show()

    if not path_to_save == "":
        figure.savefig(path_to_save)

