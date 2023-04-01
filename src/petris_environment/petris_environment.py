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


from src.scenes.scenes import GameScene, State, Scenes
from src.keyboard_controller.keyboard_controller import (move_down, move_left, 
                                                         move_right, rotate, Action)

logger = logging.getLogger(__name__)


class PetrisEnvironment(PyEnvironment):
    
    """Custom python environment for TF Agents. Extends PyEnvironment"""
    def __init__(self):
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

        self._max_actions = 25

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
            print("down")
            move_down()
        elif action == Action.MOVE_RIGHT:
            print("right")
            move_right()
        elif action == Action.MOVE_LEFT:
            print("left")
            move_left()
        elif action == Action.ROTATE:
            print("rotate")
            rotate()

        # Update the state after action
        self._state = np.squeeze(np.array(self._game_scene.tetris_map).flatten().tolist())
    
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
        
        return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action: Tensor):
        """
        Perform the given action and return the new situated that was a result of that action.
        Args:
            action (_type_): Action to perform
        """
        
        # TODO: Add line limit to end the game.
        
        if self._episode_ended:
            print("Restarting")
            return self.reset()
        if self._game_scene.game_over or self._actions_taken == self._max_actions:
            print("Game over")
            self._state = np.squeeze(np.array(self._game_scene.tetris_map).flatten().tolist())
            reward = State.full_line_no * 100
            logger.info(f"Episode Ended. Reward given: {reward}")
            self._episode_ended = True
            return ts.termination(np.array([self._state], dtype=np.int32), reward=reward)
        else:
            self.perform_action(action=action)
            self._actions_taken += 1
            self._state = np.squeeze(np.array(self._game_scene.tetris_map).flatten().tolist())
            # NOTE: We are wrapping it in [] to maintain the (1, 200) 
            # NOTE: shape that is specified in the observation spec.
            return ts.transition(np.array([self._state], dtype=np.int32), reward=0, discount=1.0)