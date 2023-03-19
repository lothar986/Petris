"""Script containing the random petris agent."""

import logging
from typing import List

import pygame
import tensorflow as tf
from pygame.time import Clock
from pygame.surface import Surface
from pygame.event import Event
from tf_agents.environments.py_environment import PyEnvironment

from src.scenes.scenes import GameScene, Scenes, TitleScene
from src.game_runner.game_runner import render_active_scene

logger = logging.getLogger(__name__) 
   

def play_random_agent(env: PyEnvironment, main_screen: Surface, clock: Clock, speed: int, num_episodes: int = 5) -> None:
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
        pygame.display.set_caption(f"{env.agent_name} Agent - Episode {episode}")
        
        game_scene = GameScene()
        Scenes.active_scene = game_scene
        
        env.reset()
        
        events: List[Event] = []
        while Scenes.active_scene.process_input(events=events):
            events = pygame.event.get()

            # [1] == (1, )
            random_action = tf.random.uniform([1], 0, 5, dtype=tf.int32)
            
            env.step(action=random_action)
            
            # Press escape to stop the entire game.            
            for event in events:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    logger.info("Stopping Agent...")
                    return
            
            render_active_scene(main_screen=main_screen, clock=clock, speed=speed)
            
            # If it switches to the title scene that means the game episode is over.
            # Recreate GameScene and run the next episode.
            if isinstance(Scenes.active_scene, TitleScene):
                logger.info("End of Episode %s", episode)
                break
