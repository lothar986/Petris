"""Script containing the code for the PPO petris Agent"""

from __future__ import absolute_import, division, print_function

import logging
from typing import List

from tf_agents.agents.ppo.ppo_agent import PPOAgent
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.utils.common import function

import pygame
import tensorflow as tf
from tensorflow import keras
from pygame.time import Clock
from pygame.surface import Surface
from pygame.event import Event
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.environments.utils import validate_py_environment


from src.petris_environment.petris_environment import PetrisEnvironment
from src.scenes.scenes import GameScene, Scenes, TitleScene
from src.game_runner.game_runner import render_active_scene

logger = logging.getLogger(__name__) 


def create_ppo(env: TFPyEnvironment, actor_network: ActorDistributionRnnNetwork, value_network: ValueRnnNetwork) -> PPOAgent:

    agent = PPOAgent(
        env.time_step_spec(),
        env.action_spec(),
        optimizer=keras.optimizers.Adam(learning_rate=0.2),
        actor_net=actor_network,
        value_net=value_network,
        train_step_counter=tf.Variable(0)
    )
    
    agent.initialize()
    
    agent.train = function(agent.train)
    
    return agent

def compute_avg_return(eval_env: TFPyEnvironment, batch_size: int, policy: TFPolicy, epocs: int):
    total_return = 0.0

    for _ in range(epocs):
        policy_state = policy.get_initial_state(batch_size)
        time_step = eval_env.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action, policy_state, info = policy.action(time_step, policy_state)
            time_step = eval_env.step(action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / epocs
    return avg_return.numpy()[0]

def train_ppo(main_screen: Surface, clock: Clock, speed: int, episodes: int = 5, batch_size: int = 1, log_interval: int = 200, eval_interval: int = 1000, steps_per_iteration: int = 10) -> None:


    train_env = TFPyEnvironment(environment=PetrisEnvironment())
    eval_env = TFPyEnvironment(environment=PetrisEnvironment())

    actor_network = ActorDistributionRnnNetwork(
        input_tensor_spec=train_env.observation_spec(),
        output_tensor_spec=train_env.action_spec(),
        lstm_size=(64,),
        activation_fn='gelu'
    )

    value_network = ValueRnnNetwork(
        input_tensor_spec=train_env.observation_spec(),
        activation_fn='gelu'
    )

    agent = create_ppo(train_env, actor_network, value_network)

    replay_buffer = TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=100000
    )

    dataset = replay_buffer.as_dataset(
        sample_batch_size=batch_size,
        num_steps=5,
        num_parallel_calls=4
    ).prefetch(4)
    iteration = iter(dataset)

    collect_driver = DynamicStepDriver(
        train_env,
        agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=steps_per_iteration
    )

    collect_driver.run =function(collect_driver.run)
    agent.train = function(agent.train)

    average_returns = []
    train_loss = 0
    time_step = train_env.reset()
    policy_state = agent.collect_policy.get_initial_state(train_env.batch_size)


    for i in range(episodes):
        print(f'Episode {i}\n')

        time_step, policy_state = collect_driver.run(
            time_step=time_step,
            policy_state=policy_state
        )

        for j in range(steps_per_iteration):
            experience, _ = next(iteration)
            train_loss = agent.train(experience)
            print(f"Training loss for iteration {j}: {train_loss.loss}")
        replay_buffer.clear()
        print("Done with iteration")
        if i % 1 == 0:
            avg_return = compute_avg_return(eval_env, batch_size, agent.policy, 1)
            average_returns.append(avg_return)
            print('\nIteration = {}\nLoss = {}\nAverage Return = {}'.format(i, train_loss.loss, avg_return))

# keyboard_events = pygame.event.get()
        # # Press escape to stop the entire game.

        # for event in keyboard_events:
        #     print(event)
        #     if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        #         logger.info("Stopping Agent...")
        #         return
        
        # render_active_scene(main_screen=main_screen, clock=clock, speed=speed)

def play_ppo_agent(env: TFPyEnvironment, main_screen: Surface, clock: Clock, speed: int, num_episodes: int = 5) -> None:
    """
    Runs multiple episodes the game scene for the agent to run.
    
    NOTE: Player an interfere the agent by pressing the keys.

    Args:
        env (PyEnvironment): _description_
        main_screen (Surface): _description_
        clock (Clock): _description_
        speed (int): _description_
    """


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
    