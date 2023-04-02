"""Script containing the code for the REINFORCE petris Agent"""

from __future__ import absolute_import, division, print_function

import logging
from typing import List

import tensorflow as tf
import numpy as np
import reverb
import pygame

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import py_driver
from pygame.time import Clock
from pygame.surface import Surface
from pygame.event import Event
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from src.custom_driver.reinforce_driver import ReinforceDriver
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

    table = reverb.Table(
        table_name,
        max_size=replay_buffer_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature
    )

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=table_name,
        sequence_length=None, 
        local_server=reverb_server
    )

    rb_observer = reverb_utils.ReverbAddEpisodeObserver(
        replay_buffer.py_client,
        table_name,
        replay_buffer_length
    )

    return replay_buffer, rb_observer

def collect_episode(env: PetrisEnvironment, policy, rb_observer, num_episodes, main_screen, clock, speed):
    driver = ReinforceDriver(
        env, 
        py_tf_eager_policy.PyTFEagerPolicy(
            policy, use_tf_function=True
        ),
        [rb_observer],
        max_episodes=num_episodes
    )
    initial_time_step = env.reset()
    driver.run(main_screen, clock, speed, initial_time_step)

# Metrics and evaluation function
def compute_avg_return(env: TFPyEnvironment, policy, num_episodes=1):

    total_return = 0.0

    for _ in range(num_episodes):

        time_step = env.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            logger.info("Manual steps (avg return)")
            time_step = env.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def create_reinforce(env: TFPyEnvironment) -> reinforce_agent.ReinforceAgent:
    logger.info("Creating agent")
    # Actor network 
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        env.observation_spec(),
        env.action_spec(),
        fc_layer_params=(100,)
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

def train_reinforce(main_screen: Surface, clock: Clock, speed: int, epochs: int = 1, log_interval: int = 1, num_eval_episodes: int = 10, eval_interval: int = 10):
    # init environment 
    petris_environment = PetrisEnvironment()
    train_enivronment = TFPyEnvironment(environment=petris_environment)
    eval_environment = TFPyEnvironment(environment=petris_environment)

    # Init the actor network, optimizer, and agent 
    reinforce_agent = create_reinforce(env=train_enivronment)
    logger.info("Agent Created")

    # TODO: THESE POLICIES ARE UNUSED, FIND CORRECT USE
    # Policies
    eval_policy = reinforce_agent.policy
    collect_policy = reinforce_agent.collect_policy

    # Init Replay Buffer
    replay_buffer, rb_observer = create_replay_buffer(agent=reinforce_agent)
    logger.info("Replay Buffer Created")

    reinforce_agent.train = common.function(reinforce_agent.train)

    # Reset the train step
    reinforce_agent.train_step_counter.assign(0)

    # Evaluate the policy before training
    logger.info("Evaluating policy before training")
    #avg_return = compute_avg_return(eval_environment, reinforce_agent.policy, num_episodes=10)
    #returns = [avg_return]
    returns = []

    logger.info("Running for %s epochs", epochs)

    for i in range(epochs):
        logger.info("Running Epoch: %s", i)

        # Save episodes to the replay buffer
        collect_episode(petris_environment, reinforce_agent.collect_policy, rb_observer=rb_observer, num_episodes=1, main_screen=main_screen, clock=clock, speed=speed)

        # Update the agent's network using the buffer data
        iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
        trajectories, _ = next(iterator)
        logger.info("We are here")
        train_loss = reinforce_agent.train(experience=trajectories)
        logger.info("Agent trained")

        replay_buffer.clear()

        # Keeps track of how many times the agent has been trained
        step = reinforce_agent.train_step_counter.numpy()
        print(f"Train step counter: {step}")

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            print("Reached eval interval")
            avg_return = compute_avg_return(eval_environment, reinforce_agent.policy, num_eval_episodes)
            print('step = {0}: loss = {1}'.format(step, avg_return))
            returns.append(avg_return)


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