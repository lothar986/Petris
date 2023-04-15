"""Script containing the code for the PPO petris Agent"""

from __future__ import absolute_import, division, print_function

import logging, time, os
from typing import List

from tf_agents.agents.ppo.ppo_agent import PPOAgent
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.trajectories import Trajectory
from tf_agents.utils.common import function
from tf_agents.utils import nest_utils
from tf_agents.specs import tensor_spec

import pygame
import reverb
import tensorflow as tf
from tensorflow import keras
from pygame.time import Clock
from pygame.surface import Surface
from pygame.event import Event
from pandas import DataFrame,concat
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.environments.utils import validate_py_environment
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.utils.common import Checkpointer

from src.params.parameters import Parameters
from src.metrics.save_metrics import plot_one, save_json, save_results
from src.custom_driver.petris_driver import PetrisDriver
from src.metrics.metrics import Metrics
from src.petris_environment.petris_environment import PetrisEnvironment
from src.scenes.scenes import GameScene, Scenes, TitleScene
from src.game_runner.game_runner import render_active_scene

logger = logging.getLogger(__name__) 

def create_replay_buffer(agent: PPOAgent, replay_buffer_length: int = 10000):
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

def collect_episode(env: PetrisEnvironment, policy, rb_observer, parameters, main_screen, clock, speed, epoch, iteration, agent):
    driver = PetrisDriver(
        env, 
        py_tf_eager_policy.PyTFEagerPolicy(
            epsilon_greedy_policy.EpsilonGreedyPolicy(
                policy=policy,
                epsilon=parameters.epsilon
            ), use_tf_function=True
        ),
        [rb_observer],
        max_episodes=parameters.collect_num_episodes,
        agent=agent
    )
    time_step = env.reset()
    policy_state = policy.get_initial_state(batch_size=1)
    driver.run(main_screen, clock, speed, epoch, iteration, time_step, policy_state)

def create_ppo(env: TFPyEnvironment, parameters: Parameters) -> PPOAgent:
    actor_network = ActorDistributionRnnNetwork(
        input_tensor_spec=env.observation_spec(),
        output_tensor_spec=env.action_spec(),
        lstm_size=tuple(int(x) for x in parameters.actor.ltsm_size),
        input_fc_layer_params=tuple(int(x) for x in parameters.actor.input_layers),
        output_fc_layer_params=tuple(int(x) for x in parameters.actor.output_layers),
        activation_fn=parameters.actor.activation
    )

    value_network = ValueRnnNetwork(
        input_tensor_spec=env.observation_spec(),
        lstm_size=tuple(int(x) for x in parameters.value.ltsm_size),
        input_fc_layer_params=tuple(int(x) for x in parameters.value.input_layers),
        output_fc_layer_params=tuple(int(x) for x in parameters.value.input_layers),
        activation_fn=parameters.value.activation
    )

    agent = PPOAgent(
        env.time_step_spec(),
        env.action_spec(),
        optimizer=keras.optimizers.Adam(learning_rate=parameters.learning_rate),
        actor_net=actor_network,
        value_net=value_network,
        train_step_counter=tf.Variable(0)    
        )
    
    agent.initialize()
    
    agent.train = function(agent.train)
    
    return agent

def compute_avg_return(env: TFPyEnvironment, policy: TFPolicy, num_episodes: int, main_screen: Surface, clock: Clock, speed: int, epoch: int, iteration: int, agent: PPOAgent) -> float:
    total_return = 0.0

    for _ in range(num_episodes):
        pygame.display.set_caption(f"EVALUATION | {agent} | Iteration {iteration+1} | Epoch {epoch+1} | Episode {_+1}")
        keyboard_events : List[Event] = []
        time_step = env.reset()
        episode_return = 0.0
        policy_state = policy.get_initial_state(env.batch_size)

        while not time_step.is_last():
            Scenes.active_scene.process_input(events=keyboard_events)
            keyboard_events = pygame.event.get()
            action_step = policy.action(time_step=time_step,policy_state=policy_state)
            time_step = env.step(action_step.action)
            episode_return += time_step.reward
            render_active_scene(main_screen=main_screen, clock=clock, speed=speed)
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def train_ppo(main_screen: Surface, clock: Clock, speed: int, metrics: Metrics, parameters: Parameters, iteration: int) -> None:

    env = PetrisEnvironment(parameters=parameters)
    train_env = TFPyEnvironment(environment=env)
    eval_env = TFPyEnvironment(environment=env)

    result_parameters = parameters
    parameters = parameters.params.agent

    agent = create_ppo(train_env, parameters)

    replay_buffer, rb_observer = create_replay_buffer(agent)

    iterater = iter(replay_buffer.as_dataset(sample_batch_size=1))

    avg_return =  compute_avg_return(eval_env, agent.policy, parameters.num_eval_episodes, main_screen, clock, speed, 0, iteration, "PPO")
    loss = 0.00
    output_data = DataFrame(data=[[0,avg_return,loss,0]], columns=['epoch','return','loss','lines_cleared'])

    for i in range(parameters.epochs):
        logger.info(f'Episode {i}\n')

        collect_episode(
            env, 
            agent.collect_policy, 
            rb_observer=rb_observer, 
            parameters=parameters, 
            main_screen=main_screen, 
            clock=clock, 
            speed=speed, 
            epoch=i, 
            iteration=iteration, 
            agent="PPO"
        )

        experience, _ = next(iterater)
        start_time = time.time()
        train_loss = agent.train(experience)
        elapsed_time = time.time() - start_time
        min, sec = divmod(elapsed_time, 60)
        logger.info(f"Training loss for epoch {i}: {train_loss.loss} | Time taken: {int(min)}:{int(sec)}")

        replay_buffer.clear()

        step = agent.train_step_counter.numpy()

        loss = train_loss.loss.numpy()

        if step % parameters.eval_interval == 0 and step != 0:
            avg_return = compute_avg_return(eval_env, agent.policy, parameters.num_eval_episodes, main_screen, clock, speed, i, iteration, "PPO")
            logger.info('Iteration = {} | Loss = {} | Average Return = {}'.format(i, loss, avg_return))

        append = DataFrame(data=[[i+1,avg_return,loss,metrics.metrics_observer().lines_placed]], columns=['epoch','return','loss','lines_cleared'])
        output_data = concat([output_data,append], ignore_index=True)
    return output_data