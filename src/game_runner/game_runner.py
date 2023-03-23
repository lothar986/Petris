"""Collection of functions that will start the Petris game"""


import pygame
from pygame.time import Clock
from pygame.surface import Surface
from src.scenes.scenes import Scenes


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
