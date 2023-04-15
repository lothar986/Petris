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
from src.params.parameters import Parameters, get_nested_value
from src.petris_environment.petris_environment import PetrisEnvironment
from src.game_runner.game_runner import play_game
from src.agents.random_agent import play_random_agent
from src.agents.dqn import play_dqn_agent
from src.agents.ppo import train_ppo
from src.agents.reinforce_agent import train_reinforce
from src.metrics.metrics import Metrics
from tf_agents.environments.tf_py_environment import TFPyEnvironment

logger = logging.getLogger(__name__)

PETRIS_LOG_FILE = "petris.log"
PETRIS_LOG_DIR = "logs"
PETRIS_LOG_PATH = paths.BASE_DIR / PETRIS_LOG_DIR / PETRIS_LOG_FILE


def main(speed: int, paramFile: Optional[str] = None , debug: bool = False) -> int:
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
        logger.info("Args: (speed=%s, parameters=%s)", speed, paramFile)
        
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
        if paramFile and os.path.isfile(paramFile):
            parameters = Parameters(paramFile)
            metrics = Metrics(parameters=parameters)
            if "index" not in parameters.iterations:
                parameters.iterations.index = None
            agent = parameters.agent
            iterations = parameters.iterations.num_iterations if parameters.iterations and parameters.iterations.num_iterations > 0 else 1
            for iteration in range(iterations):
                train_results = None
                if agent and agent.lower() == "random":
                    play_random_agent(env=PetrisEnvironment(parameters=parameters), 
                                    main_screen=main_screen, 
                                    clock=clock, 
                                    speed=speed, 
                                    num_episodes=parameters.agent.epoch)
                elif agent and agent.lower() == "dqn":
                    tf_env = TFPyEnvironment(environment=PetrisEnvironment(parameters=parameters))
                    train_results = play_dqn_agent(env=tf_env, main_screen=main_screen, clock=clock, speed=speed)
                elif agent and agent.lower() == "reinforce":
                    logger.info("Training Reinforce")
                    train_results = train_reinforce(main_screen=main_screen, clock=clock, speed=speed, parameters=parameters, metrics=metrics, iteration=iteration)
                elif agent and agent.lower() == "ppo":
                    logger.info("Training PPO")
                    train_results = train_ppo(main_screen=main_screen, clock=clock, speed=speed, parameters=parameters, metrics=metrics, iteration=iteration)
                prev = get_nested_value(parameters.params,parameters.iterations.to_change, parameters.iterations.index)
                parameters.update_param(
                    parameters.iterations.to_change, 
                    prev + parameters.iterations.delta,
                    parameters.iterations.index
                )
                metrics.finish_iteration(train_results)
            metrics.finish_training()
        else:
            logger.info('No parameters found, playing game instead')
            play_game(main_screen=main_screen, clock=clock, speed=speed)
        pygame.quit()
    except Exception as ex:
        exit_code = 1
        logger.exception(ex)
    finally:
        logger.info("End of Petris Game! Code: %s", exit_code)
        
    return exit_code


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-s", "--speed", action="store", required=False, default=50, type=int,
                        help="The speed at which the tetris piece gets dropped. "
                        "Higher is faster. Default is 50.")
    parser.add_argument("-d", "--debug", action="store_true", required=False, default=False,
                        help="Displays the debug logs.")
    parser.add_argument("-p", "--parameters", action="store", required=False,type=str,
                        help="JSON file that contains the parameters for running the agent and enviornment.")
    
    args, _ = parser.parse_known_args()
    
    sys.exit(main(speed=args.speed, 
                  paramFile = args.parameters,
                  debug=args.debug))
