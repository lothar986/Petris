"""Module for holding functions that produce graphs and plots."""

import logging
from typing import List

import matplotlib.pyplot as plt

from src import paths

logger = logging.getLogger(__name__)

RESULTS_DIR = paths.BASE_DIR / "results"
LINE_STYLES = ["solid", "dashed", "dotted", "dashdot"]


def save_plt_as_image(title: str) -> None:
    """
    Saves the file path
    Args:
        title (str): 
    """

    title = title.lower()
    title = title.replace(' ', '_')
    file_path = RESULTS_DIR / f"{title}.png"

    if file_path.is_file():
        file_path.unlink()

    file_path.parent.mkdir(exist_ok=True)
    plt.savefig(file_path)
    plt.close()


def save_plot(x: List[int], 
              y: List[int] = None, 
              x_label: str = "", 
              y_label: str = "", 
              title: str = "", 
              label: str = "",
              linestyle: str = "solid") -> None:
    
    if title:
        plt.title(title)

    if x_label:
        plt.xlabel(x_label)
    
    if y_label:
        plt.ylabel(y_label)

    if label:
        plt.legend(loc="upper left")

        if y:
            plt.plot(x, y, label=label, linestyle=linestyle)
        else:
            plt.plot(x, label=label, linestyle=linestyle)
    else:

        if y:
            plt.plot(x, y)
        else:
            plt.plot(x)

    save_plot(title)


def save_many_plots(x: List[List[int]], 
                    y: List[List[int]],
                    title: str, 
                    labels: List[str], 
                    x_label: str = "", 
                    y_label: str = "") -> None:
    """
    Plots many datapoints in  a line graph
    Args:
        data (List[List[int]]): _description_
        title (str): _description_
        x_label (str, optional): _description_. Defaults to "".
        y_label (str, optional): _description_. Defaults to "".
    """

    if len(x) != len(labels):
        logger.error("Unable to plot data make there is an equal amount labels for the data list.")
        return
    elif len(x) != len(y):
        logger.error("")

    num_styles = len(LINE_STYLES)

    for idx, datum in enumerate(data):
        #save_plot(x=x, y=y, x_label=x_Label, y_label=y_label, title=title,labels[])
        # plt.plot(datum, label=labels[idx], linestyle=LINE_STYLES[idx % num_styles])

    plt.title(title)
    plt.legend(loc="upper left")
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    save_plt(title=title)
