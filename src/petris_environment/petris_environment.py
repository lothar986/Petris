"""
Main Petris python environmnent for the agent
Courtesy of TensorFlow Docs:
- https://www.tensorflow.org/agents/tutorials/2_environments_tutorial
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
from typing import List, Optional

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from tf_agents.environments import py_environment
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments import tf_environment
from tf_agents.environments import utils

from tf_agents.specs import array_spec
from tf_agents.specs.array_spec import BoundedArraySpec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories.time_step import TimeStep

from src.params.parameters import Parameters
from src.scenes.scenes import GameScene, State, Scenes
from src.keyboard_controller.keyboard_controller import (move_down, move_left, 
                                                         move_right, rotate, Action)

logger = logging.getLogger(__name__)


class PetrisEnvironment(PyEnvironment):
    """Custom python environment for TF Agents. Extends PyEnvironment"""
    def __init__(self, parameters: Parameters):
        super().__init__()
        self._game_scene: GameScene = GameScene()
        
        # Specify action range: [ 0: Down, 1: Left, 2: Right, 3: Rotate]
        self._action_spec: BoundedArraySpec = BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name="action"
        )

        # Specify observation
        self._observation_spec: BoundedArraySpec = BoundedArraySpec(
            shape=(1, 200), dtype=np.int32, minimum=0, name="observation"
        )
        
        # State will represent the current state of the tetris map
        # Orignal Shape: (20, 10)
        # New shape: (200, )
        self._state: np.ndarray = np.squeeze(np.array(self._game_scene.tetris_map).flatten().tolist())
        
        # Flag for a game ends. Normally happens when the agent loses.
        self._episode_ended: bool = False

        # Number of actions that has been taken. Used to set a hard stop for the game ()
        self._actions_taken: int = 0

        self._prev_lines_cleared = 0

        self._parameters = parameters.params.environment

        self._max_actions = self._parameters.max_actions

        self._total_reward = 0

        self._collision_detected = False

        self._down_reward = 0
        
    def collision_detected(self) -> bool:
        return self._collision_detected

    def action_spec(self) -> BoundedArraySpec:
        return self._action_spec

    def observation_spec(self) -> BoundedArraySpec:
        return self._observation_spec
    
    def perform_action(self, action: Tensor) -> None:
        """_summary_
        Args:
            action (Tensor): _description_
        Returns:
            bool: _description_
        """

        if action == Action.MOVE_DOWN:
            move_down()
        elif action == Action.MOVE_RIGHT:
            move_right()
        elif action == Action.MOVE_LEFT:
            move_left()
        elif action == Action.ROTATE:
            rotate()

        # Update the state after action
        self._state = np.squeeze(np.array(self._game_scene.tetris_map).flatten().tolist())

    def count_holes(self, tetris_map: np.ndarray) -> int:
        holes = 0
        for col in range(tetris_map.shape[1]):
            found_block = False
            for row in range(tetris_map.shape[0]):
                if tetris_map[row, col] != 0:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes

    def penalize_holes(self, tetris_map: np.ndarray) -> float:
        num_holes = self.count_holes(tetris_map)
        penalty = -1 * num_holes * self._parameters.holes_penalty
        return penalty

    def max_height(self, tetris_col: np.ndarray) -> int:
        if np.any(tetris_col[:] != 0):
            for i,e in enumerate(tetris_col):
                if e != 0:
                    return len(tetris_col) - i
        else:
            return 0

    def height_differences(self, tetris_map: np.ndarray) -> int:
        heights = [self.max_height(tetris_map[:, col]) for col in range(tetris_map.shape[1])]
        return sum(abs(heights[i] - heights[i - 1]) for i in range(1, len(heights)))

    def penalize_height_differences(self, tetris_map: np.ndarray) -> float:
        height_diffs = self.height_differences(tetris_map)
        penalty = -1 * height_diffs * self._parameters.height_penalty
        return penalty
    
    def reward_line_clear(self) -> float:
        num_lines = State.full_line_no - self._prev_lines_cleared
        self._prev_lines_cleared += num_lines
        return self._parameters.line_reward[num_lines-1]

    def _reset(self) -> TimeStep:
        """
        Resets the environment state for a new game
        Returns:
            TimeStep: ????
        """
        
        logger.info("Restarting Environment")
        
        State.reset_new_game()
        self._game_scene = GameScene()
        Scenes.active_scene = self._game_scene
        self._state = np.squeeze(np.array(self._game_scene.tetris_map).flatten().tolist())
        self._episode_ended = False
        self._actions_taken = 0
        self._prev_lines_cleared = 0
        self._total_reward = 0
        self._point_collected = False
        self._down_reward = 0
        
        return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action: Tensor):
        """
        Perform the given action and return the new situated that was a result of that action.
        Args:
            action (_type_): Action to perform
        """
        # TODO: Add line limit to end the game.
        
        if self._episode_ended:
            return self.reset()
        if self._game_scene.game_over:
            self._state = np.squeeze(np.array(self._game_scene.tetris_map).flatten().tolist())
            logger.info(f"Episode Ended. Reward given: {self._total_reward - self._parameters.game_over_penalty}")
            self._episode_ended = True
            return ts.termination(np.array([self._state], dtype=np.int32), reward= -self._parameters.game_over_penalty)
        else:
            # Perform action, update state
            self.perform_action(action=action)
            self._actions_taken += 1
            self._state = np.squeeze(np.array(self._game_scene.tetris_map).flatten().tolist())
            # Assign penalty if it has been placed
            penalty = 0
            reward = 0
            if (self._game_scene.collision and not self._collision_detected):
                penalty = self.penalize_holes(np.array(self._game_scene.tetris_map)) + self.penalize_height_differences(np.array(self._game_scene.tetris_map))
                self._collision_detected = True
                self._game_scene.collision = False
            elif(not self._game_scene.collision):
                self._collision_detected = False

            if(self._prev_lines_cleared != State.full_line_no):
                reward = self.reward_line_clear()
                logger.info('given line')
            else: 
                if action == Action.MOVE_DOWN and not self._game_scene.is_block_finished():
                    reward = self._parameters.block_placed_reward + self._parameters.press_down_reward if self._collision_detected else self._parameters.press_down_reward
                    self._down_reward += self._parameters.press_down_reward
                else:
                    reward = self._parameters.block_placed_reward if self._collision_detected else 0
            self._total_reward += reward + penalty
            if self._collision_detected:
                if action == Action.MOVE_DOWN and not self._game_scene.is_block_finished():
                    logger.info(f'Reward Given:({reward-self._parameters.press_down_reward} + {self._down_reward:.2f} + {penalty:.2f}) = {(reward-self._parameters.press_down_reward) + self._down_reward + penalty:.2f} | Total reward : {self._total_reward:.2f}')
                else:
                    logger.info(f'Reward Given:({reward} + {self._down_reward:.2f} + {penalty:.2f}) = {reward + penalty:.2f} | Total reward : {self._total_reward:.2f}')
                self._down_reward = 0
            return ts.transition(np.array([self._state], dtype=np.int32), reward= reward + penalty, discount=0.95)
            # NOTE: We are wrapping it in [] to maintain the (1, 200) 
            # NOTE: shape that is specified in the observation spec.
