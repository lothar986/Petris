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
    _CONTROLLER.release(Key.left)


def move_right() -> None:
    _CONTROLLER.press(Key.right)
    _CONTROLLER.release(Key.right)
    
    
def move_down() -> None:
    _CONTROLLER.press(Key.down)
    _CONTROLLER.release(Key.down)
    
    
def rotate() -> None:
    _CONTROLLER.press(Key.up)
    _CONTROLLER.release(Key.up)
    
    
def move_to_bottom() -> None:
    _CONTROLLER.press(Key.space)
    _CONTROLLER.release(Key.space)

def quit() -> None:
    _CONTROLLER.press(Key.esc)
    _CONTROLLER.release(Key.esc)
