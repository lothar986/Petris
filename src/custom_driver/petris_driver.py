from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
import logging
import inspect
from typing import List
from tf_agents.drivers import driver
from tf_agents.environments import py_environment
from tf_agents.policies import py_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
import pygame
from pygame.time import Clock
from pygame.surface import Surface
from pygame.event import Event
from src.scenes.scenes import GameScene, Scenes, TitleScene
from src.game_runner.game_runner import render_active_scene

from tf_agents.typing import types

logger = logging.getLogger(__name__)

class PetrisDriver(driver.Driver):
  """A driver that runs a python policy in a python environment."""

  def __init__(
      self,
      env: py_environment.PyEnvironment,
      policy: py_policy.PyPolicy,
      observers: Sequence[Callable[[trajectory.Trajectory], Any]],
      transition_observers: Optional[Sequence[Callable[[trajectory.Transition],
                                                       Any]]] = None,
      max_steps: Optional[types.Int] = None,
      max_episodes: Optional[types.Int] = None,
      end_episode_on_boundary: bool = True,
      agent: str = None):

    max_steps = max_steps or 0
    max_episodes = max_episodes or 0
    if max_steps < 1 and max_episodes < 1:
      raise ValueError(
          'Either `max_steps` or `max_episodes` should be greater than 0.')

    super(PetrisDriver, self).__init__(env, policy, observers, transition_observers)
    self._max_steps = max_steps or np.inf
    self._max_episodes = max_episodes or np.inf
    self._end_episode_on_boundary = end_episode_on_boundary
    self._agent = agent

  def run(
      self,
      main_screen: Surface, 
      clock: Clock, 
      speed: int,
      epoch: int,
      iteration: int,
      time_step: ts.TimeStep,
      policy_state: types.NestedArray = ()
  ) -> Tuple[ts.TimeStep, types.NestedArray]:
    num_steps = 0
    num_episodes = 0
    keyboard_events : List[Event] = []
    
    while num_steps < self._max_steps and num_episodes < self._max_episodes:
      pygame.display.set_caption(f"COLLECTION | {self._agent} | Iteration {iteration+1} | Epoch {epoch+1} | {'Episode' if self._max_episodes != np.inf else 'Step'} {num_episodes+1 if self._max_episodes != np.inf else num_steps+1}")
      Scenes.active_scene.process_input(events=keyboard_events)
      keyboard_events = pygame.event.get()

      # Press escape to stop the entire game.            
      for event in keyboard_events:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            logger.info("Stopping Agent...")
            return

      # For now we reset the policy_state for non batched envs.
      if not self.env.batched and time_step.is_first() and num_episodes > 0:
        policy_state = self._policy.get_initial_state(self.env.batch_size or 1)

      action_step = self.policy.action(time_step, policy_state)
      next_time_step = self.env.step(action_step.action)
      
      # When using observer (for the purpose of training), only the previous
      # policy_state is useful. Therefore substitube it in the PolicyStep and
      # consume it w/ the observer.
      action_step_with_previous_state = action_step._replace(state=policy_state)
      traj = trajectory.from_transition(
          time_step, action_step_with_previous_state, next_time_step)
      for observer in self._transition_observers:
        observer((time_step, action_step_with_previous_state, next_time_step))
      for observer in self.observers:
        observer_signature = inspect.signature(observer)
        if "collision_detected" in observer_signature.parameters:
            observer(traj, collision_detected=self.env._collision_detected)
        else:
            observer(traj)

      if self._end_episode_on_boundary:
        num_episodes += np.sum(traj.is_boundary())
      else:
        num_episodes += np.sum(traj.is_last())

      num_steps += np.sum(~traj.is_boundary())

      time_step = next_time_step
      policy_state = action_step.state

      render_active_scene(main_screen=main_screen, clock=clock, speed=speed)
    return time_step, policy_state