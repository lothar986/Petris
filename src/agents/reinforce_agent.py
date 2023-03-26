"""Script containing the code for the REINFORCE petris Agent"""

from __future__ import absolute_import, division, print_function

import logging
from typing import List

import tensorflow as tf

from tf_agents.agents.reinforce import reinforce_agent
from pygame.time import Clock
from pygame.surface import Surface
from tf_agents.networks import actor_distribution_network
from tf_agents.environments.tf_py_environment import TFPyEnvironment

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

    cumulative_reward = 0

    agent: reinforce_agent.ReinforceAgent = create_reinforce(env=env)
    policy = agent.policy