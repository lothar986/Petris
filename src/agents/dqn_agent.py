"""Script containing the code for the DQN petris Agent"""

import logging
from typing import List

import pygame
import numpy as np
import tensorflow as tf
from pygame.time import Clock
from pygame.surface import Surface
from pygame.event import Event
from tf_agents.environments import utils
from tf_agents.environments.py_environment import PyEnvironment

from src.scenes.scenes import GameScene, Scenes, TitleScene
from src.game_runner.game_runner import render_active_scene

logger = logging.getLogger(__name__) 

# Function to test the environment using a fixed policy 
def fixed_policy_test(env: PyEnvironment):
    # Define the possible actions that can be used in our environment 
    move_down_action = np.array(0, dtype=np.int32)
    move_right_action = np.array(1, dtype=np.int32)
    move_left_action = np.array(2, dtype=np.int32)
    rotate_action = np.array(3, dtype=np.int32)
    drop_to_bottom_action = np.array(4, dtype=np.int32)

    time_step = env.reset()
    print(time_step)
    cumulative_reward = time_step.reward

    #for _in range(4):
    #    time_step = 

    print('Final Reward = ', cumulative_reward)

def play_dqn_agent(env: PyEnvironment, main_screen: Surface, clock: Clock, speed: int, num_episodes: int = 5) -> None:
    """
    Runs multiple episodes the game scene for the agent to run.
    
    NOTE: Player an interfere the agent by pressing the keys.

    Args:
        env (PyEnvironment): _description_
        main_screen (Surface): _description_
        clock (Clock): _description_
        speed (int): _description_
    """

    # NOTE: At this point, we have already defined the environment
    # in main() in petris.py 

    # Validate our environment using a random policy for 5 eps
    utils.validate_py_environment(env, episodes=5)

    fixed_policy_test(env)
