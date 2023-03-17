"""
Main Petris python environmnent for the agent
Courtesy of TensorFlow Docs:
- https://www.tensorflow.org/agents/tutorials/2_environments_tutorial
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from typing import List

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
        
        # Specify action range: [ 0: Down, 1: Left, 2: Right, 3: Rotate, 4: Spacebar ]
        self._action_spec: BoundedArraySpec = BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=4, name="action"
        )

        # Specify observation
        self._observation_spec: BoundedArraySpec = BoundedArraySpec(
            shape=(20, 10), dtype=np.int32, minimum=0
        )
        
        # State will represent the current state of the tetris map
        self._state: List[List[int]] = [[0]*10]*20
        
        # Flag for a game ends. Normally happens when the agent loses.
        self._episode_ended: bool = False
    
    @property
    def action_spec(self) -> BoundedArraySpec:
        return self._action_spec

    @property
    def observation_spec(self) -> BoundedArraySpec:
        return self._observation_spec
    
    def _reset(self) -> TimeStep:
        """
        Resets the environment state for a new game

        Returns:
            TimeStep: ????
        """
        self._state = [[0]*10]*20
        self._episode_ended = False
        
        return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):
        """
        Perform the given action and return the new situated that was a result of that action.

        Args:
            action (_type_): Action to perform
        """
        pass
