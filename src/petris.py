"""Main launcher script for the game"""

# Built-in libs
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List

# Third party libs
import pygame
from pygame.event import Event
from pygame.time import Clock
from pygame.surface import Surface
from tf_agents.environments.py_environment import PyEnvironment

# NOTE: Going to set the paths in here before importing packages. 
# NOTE: This is important for importing packages.
sys.path.append(str(Path().parent))

# Custom packages.
from src import paths
from src.log.log import initialize_logger
from src.scenes.scenes import GameMetaData, TitleScene, Scenes, GameScene

from src.petris_environment.petris_environment import PetrisEnvironment

logger = logging.getLogger(__name__)

PETRIS_LOG_FILE = "petris.log"
PETRIS_LOG_DIR = "logs"
PETRIS_LOG_PATH = paths.BASE_DIR / PETRIS_LOG_DIR / PETRIS_LOG_FILE


def render_active_scene(main_screen: Surface, clock: Clock, speed: int) -> None:
    """_summary_

    Args:
        main_screen (Surface): _description_
        clock (Clock): _description_
        speed (int): _description_
    """
    
    Scenes.active_scene.update()
    Scenes.active_scene.render(screen=main_screen)
    clock.tick(speed)


def play_game(main_screen: Surface, clock: Clock, speed: int) -> None:
    """_summary_

    Args:
        main_screen (Surface): _description_
        clock (Clock): _description_
        speed (int): _description_
    """
    
    while Scenes.active_scene.process_input(events=pygame.event.get()):
        render_active_scene(main_screen=main_screen, clock=clock, speed=speed)


def play_agent(env: PyEnvironment, main_screen: Surface, clock: Clock, speed: int, num_episodes: int = 5) -> None:
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
        
        events: List[Event] = []
        while Scenes.active_scene.process_input(events=events):
            events = pygame.event.get()

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


def main(speed: int, agent: Optional[str] = None, debug: bool = False) -> int:
    """
    Main function for the game

    Args:
        speed (int): Speed at which the tetris piece gets dropped.

    Returns:
        int: Exit code
    """
    
    exit_code = 0
    
    try:
        initialize_logger(log_path=PETRIS_LOG_PATH, debug=debug)
        
        logger.info("Starting Petris Game")
        logger.info("Args: (speed=%s, agent=%s)", speed, agent)
        
        # Positioned Window on the screen
        os.environ['SDL_VIDEO_WINDOW_POS'] = "(100,100)"

        # Main game clock that allows the piece to drop.
        clock: Clock = pygame.time.Clock()
        
        screen_size = (GameMetaData.screen_width, GameMetaData.screen_height)
        main_screen: Surface = pygame.display.set_mode(size=screen_size)
        
        # Title of the window header
        pygame.display.set_caption("Petris")
        
        Scenes.titleScene = TitleScene()
        Scenes.active_scene = Scenes.titleScene
        
        logger.debug("Initialized Game Clock: %s", clock)
        logger.debug("Main Screen Built: %s", main_screen)
        logger.debug("Scene Setup: (titleScene=%s, gameScene=%s, active_scene=%s)", 
                     Scenes.titleScene, Scenes.gameScene, Scenes.active_scene)

        logger.info("Spinning up GUI")
        
        if agent:
            play_agent(env=PetrisEnvironment(agent_name=agent.upper()), main_screen=main_screen, clock=clock, speed=speed)
        else:
            play_game(main_screen=main_screen, clock=clock, speed=speed)
        
        pygame.quit()
    except Exception as ex:
        exit_code = 1
        logger.exception(ex)
    finally:
        logger.info("End of Petris Game: %s", exit_code)
        
    return exit_code


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-s", "--speed", action="store", required=False, default=50, type=int,
                        help="The speed at which the tetris piece gets dropped. "
                        "Higher is faster. Default is 50.")
    parser.add_argument("-a", "--agent", action="store", required=False, default=None, type=str,
                        help="Agent flag.")
    parser.add_argument("-d", "--debug", action="store_true", required=False, default=False,
                        help="Displays the debug logs.")
    
    args, _ = parser.parse_known_args()
    
    sys.exit(main(speed=args.speed, agent=args.agent, debug=args.debug))
