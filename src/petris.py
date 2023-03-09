"""Main launcher script for the game"""

# Built-in libs
import os
import sys
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


def main() -> int:
    
    exit_code = 0
    
    try:
        initialize_logger(log_path=PETRIS_LOG_PATH)
        
        logger.info("Starting Petris Game")
        
        # Positioned Window
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (100, 100)

        #########
        # CLOCK #
        #########
        clock = pygame.time.Clock()

        main_screen = pygame.display.set_mode((GameMetaData.screen_width, GameMetaData.screen_height))
        pygame.display.set_caption("Petris")

        Scenes.titleScene = TitleScene()
        Scenes.active_scene = Scenes.titleScene

        while True:
            Scenes.active_scene.process_input(pygame.event.get())
            Scenes.active_scene.update()
            Scenes.active_scene.render(main_screen)

            clock.tick(50)
    except Exception as ex:
        exit_code = 1
        logger.exception(ex)
    finally:
        logger.info("End of Petris Game: %s", exit_code)
        
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
