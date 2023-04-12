import pandas as pd
import numpy as np
from tf_agents.trajectories import Trajectory
from src.scenes.scenes import State


class MetricsObserver:
    def __init__(self):
        self.tetris_heatmap = np.zeros((20, 10), dtype=int)
        self.total_placements = 0
        self.lines_placed = -1

    def __call__(self, trajectory: Trajectory, collision_detected: bool = False) -> None:
        if collision_detected:
            map = np.reshape(trajectory.observation, (20, 10))
            self.tetris_heatmap += (map != 0).astype(int)
            self.total_placements += 1
        if trajectory.is_last():
            self.lines_placed = State.full_line_no

    def get_heatmap_dataframe(self) -> pd.DataFrame:
        max_placements = np.max(self.tetris_heatmap)
        avg_heatmap = self.tetris_heatmap / max_placements if max_placements > 0 else self.tetris_heatmap
        heatmap_df = pd.DataFrame(avg_heatmap, columns=[f"col_{i}" for i in range(10)])
        return heatmap_df