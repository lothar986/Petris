"""Collection of functions for recording and saving evaluation metrics and results."""

import logging
from pathlib import Path
from typing import List, Union

import numpy as np
import matplotlib.pyplot as plt

from src import paths

logger = logging.getLogger(__name__)

# All results will be stored in here.
RESULTS_DIR = paths.BASE_DIR / "results"


def save_line_graph(title: str, 
                    data: Union[np.ndarray, List], 
                    x_label: str = "", 
                    y_label: str = "") -> None:
    """_summary_

    Args:
        data (List[int]): _description_
    """

    filename = f"{title}.png"

    logger.info("Generating and Saving Line Graph: %s", (RESULTS_DIR / filename))

    # plt.plot(data)
    # plt.show(block=False)
