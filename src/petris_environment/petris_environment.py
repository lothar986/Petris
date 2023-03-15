"""Main Petris python environmnent for the agent"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils

from tf_agents.specs import array_spec
from tf_agents.specs.array_spec import BoundedArraySpec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories.time_step import TimeStep

logger = logging.getLogger(__name__)


class PetrisEnvironment(PyEnvironment):
    """Custom python environment for TF Agents. Extends PyEnvironment"""
    def __init__(self):
        
        # Specify actions
        # 0 -> Move Down
        # 1 -> Move Left
        # 2 -> Move right
        # 3 -> Rotate
        # 4 -> Shoot Down (SPACEBAR)
        self._action_spec: BoundedArraySpec = BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=4)

        # TODO: Specify observation
        # ? 1. The shapes that are currently placed at the bottom of the tetris grid
        # ? 2. Current shape the agent has
        # ? 3. The next shape the agent is going to receive
        # ? 4. Current movement speed
        # ? 5. Current position of the shape
        
        # TODO: Figure out how to use this state.
        self._state: int = 0
        
        # Flag for a game ends. Normally happens when the agent loses.
        self._episode_ended: bool = False
    
    @property
    def action_spec(self) -> BoundedArraySpec:
        return self._action_spec

    def _reset(self) -> TimeStep:
        """
        Resets the environment state for a new game

        Returns:
            TimeStep: ????
        """
        self._state = 0
        self._episode_ended = False
        
        # TODO: Why are we return this TimeStep.
        return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):
        """_summary_

        Args:
            action (_type_): _description_
        """
        pass
