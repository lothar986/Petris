from enum import Enum


class Colour(Enum):
    GAME_OVER = (111, 247, 229)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    TEAL = (0, 255, 255)
    ORANGE = (255, 165, 0)
    GOLD = (255, 215, 0)
    PURPLE = (138, 43, 226)
    BLUE = (0, 0, 255)
    LIGHT_BLUE = (122, 122, 255)
    FIREBRICK = (178, 34, 34)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)


colours_by_number = {
    1: Colour.GAME_OVER,
    2: Colour.WHITE,
    3: Colour.BLACK,
    4: Colour.TEAL,
    5: Colour.ORANGE,
    6: Colour.GOLD,
    7: Colour.PURPLE,
    8: Colour.BLUE,
    9: Colour.LIGHT_BLUE,
    10: Colour.FIREBRICK,
    11: Colour.GREEN,
    12: Colour.RED
}


def get_colour_by_number(number):
    return colours_by_number.get(number)


colours_by_name = {
    'GAME_OVER': 1,
    'WHITE': 2,
    'BLACK': 3,
    'TEAL': 4,
    'ORANGE': 5,
    'GOLD': 6,
    'PURPLE': 7,
    'BLUE': 8,
    'LIGHT_BLUE': 9,
    'FIREBRICK': 10,
    'GREEN': 11,
    'RED': 12
}


def get_colour_number_by_name(colour_str: str) -> int:
    """Gets the color value of the given colour name"""
    return colours_by_name.get(colour_str)
