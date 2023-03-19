"""Collection of all keyboard controls"""

from enum import IntEnum

from pynput.keyboard import Key, Controller


_CONTROLLER = Controller()


class Action(IntEnum):
    MOVE_DOWN = 0
    MOVE_RIGHT = 1
    MOVE_LEFT = 2
    ROTATE = 3
    MOVE_TO_BOTTOM = 4


def move_left() -> None:
    _CONTROLLER.press(Key.left)


def move_right() -> None:
    _CONTROLLER.press(Key.right)
    
    
def move_down() -> None:
    _CONTROLLER.press(Key.down)
    
    
def rotate() -> None:
    _CONTROLLER.press(Key.up)
    
    
def move_to_bottom() -> None:
    _CONTROLLER.press(Key.space)
