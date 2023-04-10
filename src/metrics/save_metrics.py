"""Module for holding functions that produce graphs and plots."""

import logging
from typing import List

import numpy as np
import json
import matplotlib.pyplot as plt

from src import paths

logger = logging.getLogger(__name__)

RESULTS_DIR = paths.BASE_DIR / "results"
LINE_STYLES = ["solid", "dashed", "dotted", "dashdot"]

def save_json(results, iteration, is_loss) -> None: 
    if (is_loss == True):
        json_result = {'iteration': iteration, 'loss': list(map(str,results))}
        file_path = RESULTS_DIR / f"loss_iteration-{iteration}.json"
    else: 
        json_result = {'iteration': iteration, 'average_return': list(map(str,results))}
        file_path = RESULTS_DIR / f"avg_return_iteration-{iteration}.json"

    with open(file_path, 'w') as f:
        json.dump(json_result, f)


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


def plot_one(x: List[int], 
             y: List[int] = None, 
             x_label: str = "", 
             y_label: str = "", 
             title: str = "", 
             label: str = "",
             linestyle: str = "solid",
             save: bool = False) -> None:
    
    #logger.info(x)
    #logger.info(y)
    
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
            
    if title and save:
        save_plt_as_image(title=title)


def save_many_plots(x: List[List[int]], 
                    title: str, 
                    labels: List[str], 
                    y: List[List[int]] = None,
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
    
    x = np.array(x)
    if y:
        y = np.array(y)

    #logger.info("X: %s", x.shape)
    #logger.info("Y: %s", y.shape)

    if len(x) != len(labels):
        logger.error("Unable to plot data make there is an equal amount labels for the data list.")
        return
    elif x and y and x.shape != y.shape:
        logger.error("X and Y must be the same dimensions")
        return

    num_styles = len(LINE_STYLES)

    if y:
        for idx in range(len(x)):
            plot_one(x=x[idx], y=y[idx], label=labels[idx], linestyle=LINE_STYLES[idx % num_styles])
    else:
        for idx, datum in enumerate(x):
            plot_one(x=datum, label=labels[idx], linestyle=LINE_STYLES[idx % num_styles])

    plt.title(title)
    plt.legend(loc="upper left")
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    save_plt_as_image(title=title)