import logging, os
from typing import List

import numpy as np
import json
import pandas as pd
from plotnine import *

from src import paths
from src.params.parameters import Parameters
from src.observers.metrics_observer import MetricsObserver 

logger = logging.getLogger(__name__)


class Metrics():
    def __init__(self, parameters: Parameters) -> None:
        self._metrics_observer = MetricsObserver()
        self._training_results_collection = []
        self._parameters = parameters
        self._iteration = 0

        os.makedirs('./results/reinforce/graphs/heatmap/', exist_ok=True)
        os.makedirs('./results/reinforce/graphs/avg_return/', exist_ok=True)
        os.makedirs('./results/reinforce/graphs/loss/', exist_ok=True)
        os.makedirs('./results/reinforce/graphs/delta/', exist_ok=True)
        os.makedirs('./results/reinforce/graphs/lines/', exist_ok=True)

    def metrics_observer(self) -> MetricsObserver:
        return self._metrics_observer

    def finish_iteration(self, iteration_data: pd.DataFrame):
        self._iteration += 1
        iteration_data['lines'] = self._metrics_observer.lines_placed
        self._training_results_collection.append(iteration_data)
        self._generate_heatmap()
        self._generate_return_plot()
        self._generate_loss_plot()
        self._metrics_observer.reset()

    def finish_training(self) -> None:
        self._save_results()
        self._generate_return_delta_plot()
        self._generate_lines_plot()
        logger.info('Finished training, all graphs generated.')
    
    def _generate_heatmap(self) -> None:
        heatmap_df = self._metrics_observer.get_heatmap_dataframe()

        # convert the dataframe to long format
        heatmap_df_long = pd.melt(heatmap_df.reset_index(), id_vars=['index'], var_name='column', value_name='value')

        # create the heatmap plot
        heatmap_plot = (ggplot(heatmap_df_long, aes(x='column', y='index', fill='value'))
        + geom_tile(color='white')
        + scale_fill_gradient(low='white', high='red', limits=[0, 1])
        + xlab('')
        + ylab('')
        + ggtitle(f'Tetris Heatmap, Iteration {self._iteration}')
        + theme(axis_text_x=element_text(angle=45, hjust=1))
        )
        ggsave(plot=heatmap_plot, filename = f'./results/{self._parameters.agent}/graphs/heatmap/{self._parameters.hash}_{self._iteration}.png')

    def _generate_return_plot(self) -> None:
        avg_return_data = self._training_results_collection[-1][self._training_results_collection[-1]['return'] != -1]
        avg_return_plot = (ggplot(avg_return_data, aes(x='epoch', y='return'))
         + geom_point(color='blue')
         + geom_smooth(color='red')
         + ggtitle(f'Avg_return per Epoch, Iteration {self._iteration}')
        )
        ggsave(avg_return_plot, f'./results/{self._parameters.agent}/graphs/avg_return/{self._parameters.hash}_{self._iteration}.png')

    def _generate_loss_plot(self) -> None:
        loss_data = self._training_results_collection[-1]
        loss_plot = (ggplot(loss_data, aes(x='epoch', y='loss'))
         + geom_point(color='blue')
         + geom_smooth(color='red')
         + ggtitle(f'Loss per Epoch, Iteration {self._iteration}')
        )
        ggsave(loss_plot, f'./results/{self._parameters.agent}/graphs/loss/{self._parameters.hash}_{self._iteration}.png')

    def _generate_return_delta_plot(self) -> None: #dont know if works
        # concatenate DataFrames into a single DataFrame
        training_results = pd.concat([df[df['return'] != -1.00].reset_index(drop=True) for df in self._training_results_collection], axis=1)

        # calculate the mean return value for each epoch
        avg_return = training_results.groupby(training_results.index // len(self._training_results_collection)).mean()['return']
        print(avg_return)
        # calculate the reward delta for each iteration
        reward_delta = avg_return.diff(axis=1).values
        print(reward_delta)
        reward_delta_df = pd.DataFrame({'epoch': range(1, len(avg_return)), 'delta': reward_delta})

        # create the reward delta plot
        return_delta_plot = (ggplot(reward_delta_df, aes(x='epoch', y='return'))
            + geom_line(color='blue')
            + ggtitle('Reward Delta per Iteration')
            )
        
        ggsave(return_delta_plot, f'./results/{self._parameters.agent}/graphs/delta/{self._parameters.hash}_{self._iteration}.png')

    def _generate_lines_plot(self) -> None:
        # combine all of the DataFrames into a single DataFrame
        df_combined = pd.concat(dfs, keys=range(len(dfs)))

        # reset the index and rename the level 0 index
        df_combined = df_combined.reset_index().rename(columns={'level_0': 'iteration'})

        # create the plot
        lines_plot = ggplot(df_combined, aes(x='iteration', y='lines', group='level_1', color='level_1')) + geom_line()
        ggsave(lines_plot, f'./results/{self._parameters.agent}/graphs/lines/{self._parameters.hash}_{self._iteration}.png')


    def _save_results(self) -> None:
        os.makedirs(f"./results/{self._parameters.agent}/output/", exist_ok=True)
        output = self._parameters.format_output()
        df_data = []
        for df in self._training_results_collection:
            df_data.append(json.loads(df.to_json(orient='records')))
        
        output.results = df_data

        fileName = f'./results/{self._parameters.agent}/output/{self._parameters.hash}_{self._iteration}.json'

        open(fileName,'w').write(json.dumps(output))

        if os.path.isfile(fileName):
            logger.info(f"Saved results of iteration {self._iteration}. Parent hash: {self._parameters.hash}") 
        else:
            logger.error("File not found, stop the program and debug!") 
        return None
     