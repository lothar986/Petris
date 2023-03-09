"""Main launcher script for the game"""

# Built-in libs
import os
import sys
import argparse
import logging
from pathlib import Path

# Third party libs
import pygame

# NOTE: Going to set the paths in here before importing packages. 
# NOTE: This is important for importing packages.
sys.path.append(str(Path().parent))

# Custom packages.
from src import paths
from src.log.log import initialize_logger
from src.scenes.scenes import GameMetaData, TitleScene, Scenes

logger = logging.getLogger(__name__)

PETRIS_LOG_FILE = "petris.log"
PETRIS_LOG_DIR = "logs"
PETRIS_LOG_PATH = paths.BASE_DIR / PETRIS_LOG_DIR / PETRIS_LOG_FILE


def main(speed: int) -> int:
    """
    Main function for the game

    Args:
        speed (int): Speed at which the tetris piece gets dropped.

    Returns:
        int: Exit code
    """
    
    exit_code = 0
    
    try:
        initialize_logger(log_path=PETRIS_LOG_PATH)
        
        logger.info("Starting Petris Game")
        logger.info("Args: (speed=%s)", speed)
        
        # Positioned Window
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (100, 100)

        # Main game clock that allows the piece to drop.
        clock = pygame.time.Clock()

        main_screen = pygame.display.set_mode((GameMetaData.screen_width, GameMetaData.screen_height))
        pygame.display.set_caption("Petris")

        Scenes.titleScene = TitleScene()
        Scenes.active_scene = Scenes.titleScene

        while True:
            Scenes.active_scene.process_input(pygame.event.get())
            Scenes.active_scene.update()
            Scenes.active_scene.render(main_screen)

            clock.tick(speed)
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
    
    args, _ = parser.parse_known_args()
    
    sys.exit(main(speed=args.speed))
