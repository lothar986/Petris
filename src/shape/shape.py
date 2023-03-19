import copy
import random
import logging
from enum import Enum
from typing import List

import pygame

from src.colour.colour import Colour

logger = logging.getLogger(__name__)

class Direction(Enum):
    RIGHT = 1
    UP = 2
    LEFT = 3
    DOWN = 4


class Shape(object):
    def __init__(self, 
                 name: str, 
                 state: Direction, 
                 colour: Colour, 
                 row_no: int, 
                 column_no: int, 
                 head: List[int], 
                 blocks: List[int]):
        logger.debug(
            "Shape(name=%s, state=%s, colour=%s, row_no=%s, column_no=%s, head=%s, blocks=%s)",
            name, state, colour, row_no, column_no, head, blocks
        )

        self.shape_name: str = name
        self.state: Direction = state
        self.colour: Colour = colour
        self.row_no: int = row_no
        self.column_no: int = column_no
        
        # Cooridinate of the head point of the 
        self.head: List[int] = head
        
        # Holds tuples or arrays of size 2 that represents the 
        # coordinates of all the blocks in the shape.
        self.blocks: List[List[int]] = blocks
        
        # Adding head to the blocks - Makes this datastructure hold four cooridinates
        self.blocks.insert(0, self.head)

    def prepare_for_rotate(self, tetris_map):
        raise NotImplementedError

    # Methods for validating movement.

    def can_rotate(self, blocks: List[int], tetris_map: List[List[int]]) -> bool:
        """
        Checks if the current shape can rotate. Gets called when the 
        player attempts to rotate the piece.

        Args:
            blocks (List[int]): _description_
            tetris_map (List[List[int]]): _description_

        Returns:
            bool: Indicating if the shape can rotate.
        """

        for block in blocks:
            if (block[0] < 0 
                or block[0] > self.row_no - 1 
                or block[1] < 0 
                or block[1] > self.column_no - 1 
                or tetris_map[block[0]][block[1]] != 0):
                return False
        return True

    def can_move_to_right(self, tetris_map: List[List[int]]) -> bool:
        """_summary_

        Args:
            tetris_map (List[List[int]]): _description_

        Returns:
            bool: _description_
        """
        for block in self.blocks:
            if block[1] >= self.column_no - 1 or tetris_map[block[0]][block[1] + 1] != 0:
                return False
        return True

    def can_move_to_left(self, tetris_map: List[List[int]]) -> bool:
        """_summary_

        Args:
            tetris_map (List[List[int]]): _description_

        Returns:
            bool: _description_
        """
        for block in self.blocks:
            if block[1] <= 0 or tetris_map[block[0]][block[1] - 1] != 0:
                return False
        return True

    def is_finished_or_collided(self, tetris_map: List[List[int]]) -> bool:
        """_summary_

        Args:
            tetris_map (List[List[int]]): _description_

        Returns:
            bool: _description_
        """

        for block in self.blocks:
            if block[0] == self.row_no - 1 or tetris_map[block[0] + 1][block[1]] != 0:
                return True
        return False

    # Movement Controllers
    
    def rotate(self, tetris_map: List[List[int]]) -> bool:
        """_summary_

        Args:
            tetris_map (List[List[int]]): _description_

        Returns:
            bool: _description_
        """
        temp_blocks, temp_direction = self.prepare_for_rotate(tetris_map)

        if self.can_rotate(temp_blocks, tetris_map):
            self.state = temp_direction
            
            logger.debug("Blocks Before: %s", self.blocks)
            self.blocks = copy.deepcopy(temp_blocks)
            self.head = self.blocks[0]
            logger.debug("Blocks After: %s", self.blocks)

            return True
        else:
            logger.debug("Cannot rotate")
            return False

    def move_left(self, tetris_map: List[List[int]]):
        """_summary_

        Args:
            tetris_map (List[List[int]]): _description_
        """
        if self.can_move_to_left(tetris_map):
            for i in self.blocks:
                i[1] -= 1
        else:
            logger.debug("Cannot move left")

    def move_right(self, tetris_map: List[List[int]]):
        """_summary_

        Args:
            tetris_map (List[List[int]]): _description_
        """
        if self.can_move_to_right(tetris_map):
            for block in self.blocks:
                block[1] += 1
        else:
            logger.debug("Cannot move right")

    def move_down(self, tetris_map: List[List[int]]):
        """_summary_

        Args:
            tetris_map (List[List[int]]): _description_
        """
        if not self.is_finished_or_collided(tetris_map):
            for block in self.blocks:
                block[0] += 1
        else:
            logger.debug("Collision")

    # TODO: Figure out what these do.
    def draw(self, screen):
        """_summary_

        Args:
            screen (_type_): _description_
        """
        for i in self.blocks:
            pygame.draw.rect(screen, self.colour.value, (50 + (i[1] * 30), 50 + (i[0] * 30), 30, 30), 2)
            pygame.draw.rect(screen, self.colour.value, (50 + (i[1] * 30) + 5, 50 + (i[0] * 30) + 5, 21, 21))

    def draw_next(self, screen, x_position):
        """_summary_

        Args:
            screen (_type_): _description_
            x_position (_type_): _description_
        """
        for i in self.blocks:
            pygame.draw.rect(screen, self.colour.value, (x_position + (i[1] * 15), 170 + (i[0] * 15), 15, 15))
            pygame.draw.rect(screen, Colour.BLACK.value, (x_position + (i[1] * 15), 170 + (i[0] * 15), 15, 15), 1)


#
# [X][ ]
# [ ][ ]
#
class SmashBoy(Shape):
    def __init__(self, row_no, column_no):
        super().__init__('SmashBoy', None, Colour.FIREBRICK, row_no, column_no, [0, 4], [[1, 4], [0, 5], [1, 5]])

#    def rotate(self, tetris_map):
#        pass

    def prepare_for_rotate(self, tetris_map):
        return self.blocks, self.state          # pass 


#
# [ ][ ][X][ ]
#
class Hero(Shape):

    def __init__(self, row_no, column_no):
        super().__init__('Hero', Direction.DOWN, Colour.GREEN, row_no, column_no, [0, 6], [[0, 4], [0, 5], [0, 7]])

    def prepare_for_rotate(self, tetris_map):
        temp_blocks = copy.deepcopy(self.blocks)
        temp_head = temp_blocks[2]

        if self.state == Direction.DOWN:
            temp_direction = Direction.UP
            temp_blocks[0] = [temp_head[0] + 1, temp_head[1]]
            temp_blocks[1] = [temp_head[0] + 2, temp_head[1]]
            temp_blocks[2] = [temp_head[0], temp_head[1]]
            temp_blocks[3] = [temp_head[0] - 1, temp_head[1]]
        else:
            temp_direction = Direction.DOWN
            temp_blocks[0] = [temp_head[0], temp_head[1] - 2]
            temp_blocks[1] = [temp_head[0], temp_head[1] - 1]
            temp_blocks[2] = [temp_head[0], temp_head[1]]
            temp_blocks[3] = [temp_head[0], temp_head[1] + 1]

        return temp_blocks, temp_direction


#
#    [X][ ]
# [ ][ ]
#
class RhodeIsland(Shape):
    def __init__(self, row_no, column_no):
        super().__init__('RhodeIsland', Direction.UP, Colour.TEAL, row_no, column_no, [0, 4], [[0, 5], [1, 4], [1, 3]])

    def prepare_for_rotate(self, tetris_map):
        temp_blocks = copy.deepcopy(self.blocks)
        temp_head = temp_blocks[0]

        if self.state == Direction.UP:
            temp_direction = Direction.DOWN
            temp_blocks[1] = [temp_head[0] - 1, temp_head[1]]
            temp_blocks[2] = [temp_head[0], temp_head[1] + 1]
            temp_blocks[3] = [temp_head[0] + 1, temp_head[1] + 1]
        else:
            temp_direction = Direction.UP
            temp_blocks[1] = [temp_head[0], temp_head[1] + 1]
            temp_blocks[2] = [temp_head[0] + 1, temp_head[1]]
            temp_blocks[3] = [temp_head[0] + 1, temp_head[1] - 1]

        return temp_blocks, temp_direction


#
# [ ][X]
#    [ ][ ]
class Cleveland(Shape):
    def __init__(self, row_no, column_no):
        super().__init__('Cleveland', Direction.UP, Colour.GOLD, row_no, column_no, [0, 4], [[0, 3], [1, 4], [1, 5]])

    def prepare_for_rotate(self, tetris_map):
        temp_blocks = copy.deepcopy(self.blocks)
        temp_head = temp_blocks[0]

        if self.state == Direction.UP:
            temp_direction = Direction.DOWN
            temp_blocks[1] = [temp_head[0] + 1, temp_head[1]]
            temp_blocks[2] = [temp_head[0], temp_head[1] + 1]
            temp_blocks[3] = [temp_head[0] - 1, temp_head[1] + 1]
        else:
            temp_direction = Direction.UP
            temp_blocks[1] = [temp_head[0], temp_head[1] - 1]
            temp_blocks[2] = [temp_head[0] + 1, temp_head[1]]
            temp_blocks[3] = [temp_head[0] + 1, temp_head[1] + 1]

        return temp_blocks, temp_direction


#
#       [ ]
# [ ][X][ ]
#
class OrangeRicky(Shape):
    def __init__(self, row_no, column_no):
        super().__init__('OrangeRicky', Direction.RIGHT, Colour.ORANGE, row_no, column_no, [1, 5], [[1, 4], [1, 6], [0, 6]])

    def prepare_for_rotate(self, tetris_map):
        temp_blocks = copy.deepcopy(self.blocks)
        temp_head = temp_blocks[0]
        temp_direction = self.state

        if temp_direction == Direction.RIGHT:
            temp_direction = Direction.UP
            temp_blocks[1] = [temp_head[0] + 1, temp_head[1]]
            temp_blocks[2] = [temp_head[0] - 1, temp_head[1]]
            temp_blocks[3] = [temp_head[0] - 1, temp_head[1] - 1]
        elif temp_direction == Direction.UP:
            temp_direction = Direction.LEFT
            temp_blocks[1] = [temp_head[0], temp_head[1] + 1]
            temp_blocks[2] = [temp_head[0], temp_head[1] - 1]
            temp_blocks[3] = [temp_head[0] + 1, temp_head[1] - 1]
        elif temp_direction == Direction.LEFT:
            temp_direction = Direction.DOWN
            temp_blocks[1] = [temp_head[0] - 1, temp_head[1]]
            temp_blocks[2] = [temp_head[0] + 1, temp_head[1]]
            temp_blocks[3] = [temp_head[0] + 1, temp_head[1] + 1]
        else:
            temp_direction = Direction.RIGHT
            temp_blocks[1] = [temp_head[0], temp_head[1] - 1]
            temp_blocks[2] = [temp_head[0], temp_head[1] + 1]
            temp_blocks[3] = [temp_head[0] - 1, temp_head[1] + 1]

        return temp_blocks, temp_direction


#
# [ ]
# [ ][X][ ]
#
class BlueRicky(Shape):
    def __init__(self, row_no, column_no):
        super().__init__('BlueRicky', Direction.RIGHT, Colour.BLUE, row_no, column_no, [1, 5], [[1, 4], [1, 6], [0, 4]])

    def prepare_for_rotate(self, tetris_map):
        temp_blocks = copy.deepcopy(self.blocks)
        temp_head = temp_blocks[0]
        temp_direction = self.state

        if temp_direction == Direction.RIGHT:
            temp_direction = Direction.UP
            temp_blocks[1] = [temp_head[0] - 1, temp_head[1]]
            temp_blocks[2] = [temp_head[0] + 1, temp_head[1]]
            temp_blocks[3] = [temp_head[0] + 1, temp_head[1] - 1]
        elif temp_direction == Direction.UP:
            temp_direction = Direction.LEFT
            temp_blocks[1] = [temp_head[0], temp_head[1] - 1]
            temp_blocks[2] = [temp_head[0], temp_head[1] + 1]
            temp_blocks[3] = [temp_head[0] + 1, temp_head[1] + 1]
        elif temp_direction == Direction.LEFT:
            temp_direction = Direction.DOWN
            temp_blocks[1] = [temp_head[0] + 1, temp_head[1]]
            temp_blocks[2] = [temp_head[0] - 1, temp_head[1]]
            temp_blocks[3] = [temp_head[0] - 1, temp_head[1] + 1]
        else:
            temp_direction = Direction.RIGHT
            temp_blocks[1] = [temp_head[0], temp_head[1] - 1]
            temp_blocks[2] = [temp_head[0], temp_head[1] + 1]
            temp_blocks[3] = [temp_head[0] - 1, temp_head[1] - 1]

        return temp_blocks, temp_direction


#
#    [ ]
# [ ][X][ ]
#
class TeeWee(Shape):
    def __init__(self, row_no, column_no):
        super().__init__('TeeWee', Direction.DOWN, Colour.PURPLE, row_no, column_no, [1, 5], [[0, 5], [1, 4], [1, 6]])

    def prepare_for_rotate(self, tetris_map):
        temp_blocks = copy.deepcopy(self.blocks)
        temp_head = temp_blocks[0]
        temp_direction = self.state
        if temp_direction == Direction.DOWN:
            temp_direction = Direction.RIGHT
            temp_blocks[1] = [temp_head[0], temp_head[1] - 1]
            temp_blocks[2] = [temp_head[0] + 1, temp_head[1]]
            temp_blocks[3] = [temp_head[0] - 1, temp_head[1]]
        elif temp_direction == Direction.RIGHT:
            temp_direction = Direction.UP
            temp_blocks[1] = [temp_head[0] + 1, temp_head[1]]
            temp_blocks[2] = [temp_head[0], temp_head[1] + 1]
            temp_blocks[3] = [temp_head[0], temp_head[1] - 1]
        elif temp_direction == Direction.UP:
            temp_direction = Direction.LEFT
            temp_blocks[1] = [temp_head[0], temp_head[1] + 1]
            temp_blocks[2] = [temp_head[0] - 1, temp_head[1]]
            temp_blocks[3] = [temp_head[0] + 1, temp_head[1]]
        else:
            temp_direction = Direction.DOWN
            temp_blocks[1] = [temp_head[0] - 1, temp_head[1]]
            temp_blocks[2] = [temp_head[0], temp_head[1] - 1]
            temp_blocks[3] = [temp_head[0], temp_head[1] + 1]

        return temp_blocks, temp_direction


def get_random_shape(map_row_no: int, map_column_no: int) -> Shape:
    """
    Gets a random shape for the player.

    Args:
        map_row_no (int): _description_
        map_column_no (int): _description_

    Returns:
        Shape: One of the child classes that inherits the Shape class.
    """
    
    shape_type = random.randrange(0, 7)
    if shape_type == 0:
        logger.debug("Creating SmashBoy Shape")
        return SmashBoy(map_row_no, map_column_no)
    if shape_type == 1:
        logger.debug("Creating Hero Shape")
        return Hero(map_row_no, map_column_no)
    if shape_type == 2:
        logger.debug("Creating BlueRicky Shape")
        return BlueRicky(map_row_no, map_column_no)
    if shape_type == 3:
        logger.debug("Creating OrangeRicky Shape")
        return OrangeRicky(map_row_no, map_column_no)
    if shape_type == 4:
        logger.debug("Creating TeeWee Shape")
        return TeeWee(map_row_no, map_column_no)
    if shape_type == 5:
        logger.debug("Creating RhodeIsland Shape")
        return RhodeIsland(map_row_no, map_column_no)
    if shape_type == 6:
        logger.debug("Creating Cleveland Shape")
        return Cleveland(map_row_no, map_column_no)
