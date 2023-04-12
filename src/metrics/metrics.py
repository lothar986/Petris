import logging, os
from typing import List

import numpy as np
import json
from pandas import DataFrame
from plotnine import ggplot, aes, geom_line

from src import paths
from src.params.parameters import Parameters
from src.observers.metrics_observer import MetricsObserver 

logger = logging.getLogger(__name__)


class Metrics():
    def __init__(self) -> None:
        self._metrics_observer = MetricsObserver()

    def metrics_observer(self) -> MetricsObserver:
        return self._metrics_observer
    
    def generate_graph():
        None
    
    def heatmap():
        None
    

    def save_results(avg_returns: list[float], losses: list[float],iteration: int, parameters: Parameters) -> None:
        os.makedirs(f"./results/{parameters.agent}", exist_ok=True)
        output = parameters.format_output()
        output.results = {
            'returns': avg_returns,
            'loss': losses
        }
        fileName = f'./results/{parameters.agent}/{parameters.hash}_{iteration}'

        open(fileName,'w').write(json.dumps(output))

        if os.path.isfile(fileName):
            logger.info(f"Saved results of iteration {iteration}. Parent hash: {parameters.hash}") 
        else:
            logger.error("File not found, stop the program and debug!") 
        return None
     