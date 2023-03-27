"""Script containing the code for the REINFORCE petris Agent"""

from __future__ import absolute_import, division, print_function

import logging
from typing import List

import tensorflow as tf
import numpy as np

from tf_agents.agents.reinforce import reinforce_agent
from pygame.time import Clock
from pygame.surface import Surface
from tf_agents.networks import actor_distribution_network
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.specs import tensor_spec

from src.scenes.scenes import GameScene, Scenes, TitleScene
from src.game_runner.game_runner import render_active_scene
from src.petris_environment.petris_environment import PetrisEnvironment

logger = logging.getLogger(__name__) 

def create_replay_buffer(agent: reinforce_agent.ReinforceAgent, replay_buffer_length: int = 100000):
    table_name = "uniform_table"

    replay_buffer_signature = tensor_spec.from_spec(
        agent.collect_data_spec
    )
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature
    )

    

# Metrics and evaluation function
def compute_avg_return(env: TFPyEnvironment, policy, num_episodes=10):

    total_return = 0.0

    for _ in range(num_episodes):

        time_step = env.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def create_reinforce(env: TFPyEnvironment) -> reinforce_agent.ReinforceAgent:
    # Actor network 
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        env.observation_spec(),
        env.action_spec(),
        fc_layer_params=100
    )

    # NOTE: .001 lr was the example used by the docs
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.2)

    # Keeps track 
    train_step_counter = tf.Variable(0)

    agent = reinforce_agent.ReinforceAgent(
        env.time_step_spec(),
        env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter
    )

    agent.initialize()

    return agent


def play_reinforce_agent(env: TFPyEnvironment, main_screen: Surface, clock: Clock, speed: int, num_episodes: int = 5) -> None:
    """
    Runs multiple episodes the game scene for the agent to run.
    
    NOTE: Player an interfere the agent by pressing the keys.

    Args:
        env (PyEnvironment): _description_
        main_screen (Surface): _description_
        clock (Clock): _description_
        speed (int): _description_
    """
    petris_environment = PetrisEnvironment()
    environment = TFPyEnvironment(environment=petris_environment)

    cumulative_reward = 0

    agent: reinforce_agent.ReinforceAgent = create_reinforce(env=environment)
    policy = agent.policy