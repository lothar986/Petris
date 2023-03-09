"""Main launcher script for the game"""

# Built-in libs
import os
import sys
from pathlib import Path

# Third party libs
import pygame

# NOTE: Going to set the paths in here before importing packages. 
# NOTE: This is important for importing packages.
sys.path.append(str(Path().parent))

# Custom packages.
from src.scenes.scenes import GameMetaData, TitleScene, Scenes

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
