"""Main launcher script for the game"""

# Built-in libs
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# NOTE: Going to set the paths in here before importing packages. 
# NOTE: This is important for importing packages.
sys.path.append(str(Path().parent))

# Third party libs
import pygame
from pygame.time import Clock
from pygame.surface import Surface

# Custom packages.
from src import paths
from src.log.log import initialize_logger
from src.scenes.scenes import GameMetaData, TitleScene, Scenes

from src.petris_environment.petris_environment import PetrisEnvironment
from src.game_runner.game_runner import play_game
from src.agents.random_agent import play_random_agent
from src.agents.dqn_agent import play_dqn_agent

logger = logging.getLogger(__name__)

PETRIS_LOG_FILE = "petris.log"
PETRIS_LOG_DIR = "logs"
PETRIS_LOG_PATH = paths.BASE_DIR / PETRIS_LOG_DIR / PETRIS_LOG_FILE


def main(speed: int, agent: Optional[str] = None, debug: bool = False, num_episodes: int = 5) -> int:
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
        
        if agent.lower() == "random":
            play_random_agent(env=PetrisEnvironment(agent_name=agent.upper()), 
                              main_screen=main_screen, 
                              clock=clock, 
                              speed=speed, 
                              num_episodes=num_episodes)
        elif agent.lower() == "dqn":
            play_dqn_agent(env=PetrisEnvironment(agent_name=agent.upper()), main_screen=main_screen, clock=clock, speed=speed)
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
    parser.add_argument("-e", "--num-episodes", action="store", required=False, default=5, type=int,
                        help="Number of episodes for the agent to run.")
    
    args, _ = parser.parse_known_args()
    
    sys.exit(main(speed=args.speed, 
                  agent=args.agent, 
                  debug=args.debug, 
                  num_episodes=args.num_episodes))
