"""Script containing the code for the DQN petris Agent"""

from __future__ import absolute_import, division, print_function

import logging
from typing import List

from tf_agents.agents.dqn import dqn_agent
# from tf_agents.drivers import py_driver
# from tf_agents.environments import suite_gym
# from tf_agents.environments import tf_py_environment
# from tf_agents.eval import metric_utils
# from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
# from tf_agents.policies import py_tf_eager_policy
# from tf_agents.policies import random_tf_policy
# from tf_agents.replay_buffers import reverb_replay_buffer
# from tf_agents.replay_buffers import reverb_utils
# from tf_agents.trajectories import trajectory
# from tf_agents.specs import tensor_spec
from tf_agents.utils import common


import pygame
import tensorflow as tf
from tensorflow import keras
from pygame.time import Clock
from pygame.surface import Surface
from pygame.event import Event
from tf_agents.environments import utils
from tf_agents.environments.tf_py_environment import TFPyEnvironment

from src.scenes.scenes import GameScene, Scenes, TitleScene
from src.game_runner.game_runner import render_active_scene

logger = logging.getLogger(__name__) 

def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))

def create_dqn(env: TFPyEnvironment) -> dqn_agent.DqnAgent:
    q_net = sequential.Sequential([
        keras.layers.Dense(
            100, 
            activation=keras.activations.relu, 
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
        ),
        tf.keras.layers.Flatten(), 
        keras.layers.Dense(4, activation="linear")
    ])
    
    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=keras.optimizers.Adam(learning_rate=0.2),
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=tf.Variable(0)
    )
    
    agent.initialize()
    
    q_net.summary()
    
    return agent


# Function to test the environment using a fixed policy 
def fixed_policy_test(env: TFPyEnvironment):
    # Define the possible actions that can be used in our environment 
    move_down_action = tf.constant([0], dtype=tf.int32)
    move_right_action = tf.constant([1], dtype=tf.int32)
    move_left_action = tf.constant([2], dtype=tf.int32)
    rotate_action = tf.constant([3], dtype=tf.int32)

    # print("DQN fixed policy current time step:", time_step)
    # cumulative_reward = time_step.reward

    time_step = env.step(move_down_action)
    print("DQN fixed policy during play:", time_step)
    # cumulative_reward += time_step.reward
        
    # print("DQN Final Reward = ", cumulative_reward)
    return time_step

def play_dqn_agent(env: TFPyEnvironment, main_screen: Surface, clock: Clock, speed: int, num_episodes: int = 5) -> None:
    """
    Runs multiple episodes the game scene for the agent to run.
    
    NOTE: Player an interfere the agent by pressing the keys.

    Args:
        env (PyEnvironment): _description_
        main_screen (Surface): _description_
        clock (Clock): _description_
        speed (int): _description_
    """

    cumulative_reward = 0
    
    agent: dqn_agent.DqnAgent = create_dqn(env=env)
    policy = agent.policy


    # Runs multiple games without quiting the pygame
    for episode in range(1, num_episodes + 1):
        logger.info("Starting Episode %s", episode)
        
        # Display episode
        pygame.display.set_caption(f"Agent - Episode {episode}")
        
        time_step = env.reset()
        
        keyboard_events: List[Event] = []
        while not time_step.is_last():
            keyboard_events = pygame.event.get() 
            Scenes.active_scene.process_input(events=keyboard_events)
            
            # Press escape to stop the entire game.            
            for event in keyboard_events:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    logger.info("Stopping Agent...")
                    return
            
            render_active_scene(main_screen=main_screen, clock=clock, speed=speed)
            
            # NOTE: At this point, we have already defined the environment
            # in main() in petris.py 

            # Validate our environment using a random policy for 5 eps
            action = policy.action(time_step=time_step)
            env.step(action=action)
            cumulative_reward += time_step.reward
            
            # If it switches to the title scene that means the game episode is over.
            # Recreate GameScene and run the next episode.
            if isinstance(Scenes.active_scene, TitleScene):
                logger.info("End of Episode %s", episode)
                break
    
    logger.info("Cumulative Reward: %s", cumulative_reward)
    