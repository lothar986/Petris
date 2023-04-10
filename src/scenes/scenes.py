import logging
from typing import List

import pygame
from pygame import mixer
from pygame.event import Event
from pygame.surface import Surface

from src import paths
from src.shape.shape import get_random_shape, Shape
from src.colour.colour import Colour, get_colour_by_number, get_colour_number_by_name

logger = logging.getLogger(__name__)

# Paths
RESOURCES_PATH = paths.BASE_DIR / "resources"

# Audio files
AUDIO_DIR = RESOURCES_PATH / "audio"
BACKGROUND_WAV_PATH = AUDIO_DIR / "background.wav"
CAN_ROTATE_WAV_PATH = AUDIO_DIR / "can_rotate.wav"
TILT_ROTATE_WAV_PATH = AUDIO_DIR / "tilt_rotate.wav"
SPEED_MODE_SOUND_WAV_PATH = AUDIO_DIR / "speed_mode.wav"

# Font library
TETRIS_FONTS_TFF_PATH = RESOURCES_PATH / "fonts" / "tetris_font.ttf"

# Init
pygame.init()

# Sound and music mixer
mixer.music.load(BACKGROUND_WAV_PATH)
mixer.music.set_volume(0.2)
mixer.music.play(-1)

rotate_sound = mixer.Sound(CAN_ROTATE_WAV_PATH)
rotate_sound.set_volume(1)

tilt_rotate_sound = mixer.Sound(TILT_ROTATE_WAV_PATH)
tilt_rotate_sound.set_volume(1)

speed_mode_sound = mixer.Sound(SPEED_MODE_SOUND_WAV_PATH)
speed_mode_sound.set_volume(1)


class GameMetaData(object):
    """
    Collection of all assets of that make up the entire game's meta data.
        - Action Sound
        - Game Music
        - Screen Dimensions
        - Fonts
    """

    font_type = TETRIS_FONTS_TFF_PATH
    
    # Game map or grid dimensions: (20x10)
    map_row_no = 20
    map_column_no = 10
    
    screen_width = map_column_no * 30 + 250
    screen_height = map_row_no * 30 + 100
    screen_center_width = int(screen_width / 2)
    screen_center_height = int(screen_height / 2)
    width_offset = 50
    height_offset = 50
    score_window_pos = 30 * map_column_no + width_offset + 30
    score_window_text_pos = 30 * map_column_no + width_offset + 50


class Scenes(object):
    titleScene = None           # type: TitleScene
    gameScene = None            # type: GameScene
    active_scene = None         # type: TitleScene or GameScene


class State(object):
    """Game state. Keeps track of scores and levels."""    
    level: int = 1
    score: int = 0
    full_line_no: int = 0

    @staticmethod
    def reset_new_game():
        
        logger.info("Initializing Game")
        
        State.score = 0
        State.full_line_no = 0
        State.level = 1


class SceneBase:
    """Main window app scene base superclass."""
    def __init__(self):
        self.next = self
        self.score_font = pygame.font.Font(GameMetaData.font_type, 18)
        self.full_line_font = pygame.font.Font(GameMetaData.font_type, 18)
        self.level_font = pygame.font.Font(GameMetaData.font_type, 18)
        self.next_font = pygame.font.Font(GameMetaData.font_type, 16)

    def process_input(self, events: List[Event]):
        raise NotImplementedError("Uh-oh, you didn't override this (process_input) in the child class")

    def update(self):
        raise NotImplementedError("Uh-oh, you didn't override this (update) in the child class")

    def render(self, screen):
        raise NotImplementedError("Uh-oh, you didn't override this (render) in the child class")

    def draw_score_area(self, main_screen: Surface):
        """
        Draws the score area section
        
        Args:
            main_screen: Surface or sandbox of the main window
        
        """
        
        pygame.draw.rect(main_screen, Colour.FIREBRICK.value, (GameMetaData.score_window_pos, 50, 155, 85), 1)
        
        # Score label
        score_text = self.score_font.render("Score: " + str(State.score), True, Colour.WHITE.value)
        
        # Lines label
        full_line_text = self.full_line_font.render("Lines: " + str(State.full_line_no), True,
                                                    Colour.WHITE.value)
        
        # Level label
        level_text = self.level_font.render("Level: " + str(State.level), True, Colour.WHITE.value)

        # Draw and display the rectangular score section on the main window
        main_screen.blit(score_text, (GameMetaData.score_window_text_pos, 60))
        main_screen.blit(full_line_text, (GameMetaData.score_window_text_pos, 85))
        main_screen.blit(level_text, (GameMetaData.score_window_text_pos, 110))

        # This is the 'Next' section where it displays the next shape
        pygame.draw.rect(main_screen, Colour.FIREBRICK.value, (GameMetaData.score_window_pos, 140, 155, 80), 1)
        
        # Next label
        next_text = self.next_font.render('Next: ', True, Colour.WHITE.value)
        
        # Draw the 'Next' rectangle section on the main window
        main_screen.blit(next_text, (GameMetaData.score_window_text_pos - 15, 145))

    @staticmethod
    def draw_area_grid(main_screen):
        """
        Draws the main playing grid

        Args:
            main_screen: Surface or sandbox of the main window
        """
        for row_no in range(0, GameMetaData.map_row_no + 1):
            pygame.draw.line(main_screen, Colour.LIGHT_BLUE.value, (GameMetaData.width_offset, 50 + (row_no * 30)),
                         (30 * GameMetaData.map_column_no + GameMetaData.width_offset, 50 + (row_no * 30)), 1)
            if row_no < GameMetaData.map_column_no + 1:
                pygame.draw.line(main_screen, Colour.LIGHT_BLUE.value, (GameMetaData.width_offset + (row_no * 30),
                                                                    GameMetaData.height_offset),
                             (GameMetaData.width_offset + (row_no * 30),
                              30 * GameMetaData.map_row_no + GameMetaData.height_offset))

    def switch_to_scene(self, next_scene):
        self.next = next_scene


class TitleScene(SceneBase):
    """Main/pause menu scene."""
    def __init__(self):
        SceneBase.__init__(self)
        self._is_continue: bool = False
        self._is_game_over: bool = False
        self.options: int = 0 if self.is_continue else 1
        self.default = Colour.WHITE.value
        self.selected = Colour.RED.value
        self.continue_font = pygame.font.Font(GameMetaData.font_type, 36)
        self.new_game_font = pygame.font.Font(GameMetaData.font_type, 36)
        self.options_font = pygame.font.Font(GameMetaData.font_type, 36)
        self.exit_game_font = pygame.font.Font(GameMetaData.font_type, 36)

    @property
    def is_continue(self) -> bool:
        return self._is_continue

    @is_continue.setter
    def is_continue(self, is_continue):
        self._is_continue = is_continue
        self.options = 0

    @property
    def is_game_over(self):
        return self._is_game_over

    @is_game_over.setter
    def is_game_over(self, is_game_over):
        self._is_game_over = is_game_over
        self._is_continue = False
        self.options = 1

    def process_input(self, events: List[Event]) -> bool:
        """
        Processes all keys that were pressed in main or pause menu.

        Args:
            events (List): List of event objects of the key pressed. Also detects mouseOver event.
                           This list can hold more than one item if the user presses the keys quickly
        """
        for event in events:
            
            # Press down action by the user.
            if event.type == pygame.KEYDOWN:
                
                # Moves down one item in the menu
                if event.key == pygame.K_DOWN:
                    self.options += 1
                    if self.options > 3:
                        self.options = 0 if self.is_continue else 1
                
                # Moves up one item in the menu
                if event.key == pygame.K_UP:
                    self.options -= 1
                    lower_limit = 0 if self.is_continue else 1
                    if self.options < lower_limit:
                        self.options = 3
                
                # Pauses the game when the user presses the escape key.
                if event.key == pygame.K_ESCAPE:    
                    # Resumes game from menu
                    if self.is_continue:
                        logger.info("Resuming game..")
                        Scenes.active_scene = Scenes.gameScene
                
                if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    # Resumes game from main menu
                    if self.options == 0:
                        logger.info("Resuming game..")
                        Scenes.active_scene = Scenes.gameScene

                    # Starts new game
                    if self.options == 1:
                        
                        logger.info("Starting new game...")
                        
                        self._is_game_over = False
                        State.reset_new_game()
                        Scenes.gameScene = GameScene()
                        Scenes.active_scene = Scenes.gameScene

                    # Opens options menu when the user selects the "Options" button
                    if self.options == 2:
                        pass
                    
                    # Quits the game when the user selects the "Exit" button
                    if self.options == 3:
                        return quit_game()
            
            # Quits the game when
            if event.type == pygame.QUIT:
                return quit_game()
                
        return True

    def update(self):
        pass

    def render(self, screen: Surface):
        if not self.is_game_over:
            screen.fill(Colour.BLACK.value)
            self.draw_score_area(screen)

        SceneBase.draw_area_grid(screen)
        new_game_text = self.new_game_font.render("NEW GAME", True,
                                                  self.selected if self.options == 1 else self.default)
        options_text = self.options_font.render("OPTIONS", True,
                                                self.selected if self.options == 2 else self.default)
        exit_game_text = self.exit_game_font.render("EXIT", True,
                                                    self.selected if self.options == 3 else self.default)

        menu_background = pygame.Rect((0, 0), (250, 250))
        menu_rect = options_text.get_rect(center=(GameMetaData.screen_center_width,
                                                  GameMetaData.screen_center_height))
        menu_offset = 25 if self.is_continue else 0
        menu_background.center = (menu_rect.width / 2 + menu_rect.x,
                                  (menu_rect.height / 2 + menu_rect.y) - menu_offset)
        pygame.draw.rect(screen, Colour.BLACK.value, menu_background, 0)
        pygame.draw.rect(screen, Colour.WHITE.value, menu_background, 1)

        if self.is_game_over:
            game_over_font = pygame.font.Font(GameMetaData.font_type, 72)
            game_over_text = game_over_font.render("GAME OVER", True, Colour.RED.value)
            screen.blit(game_over_text,
                        game_over_text.get_rect(center=(GameMetaData.screen_center_width,
                                                        GameMetaData.screen_center_height - 160)))

        if self.is_continue:
            continue_game_text = self.continue_font.render("CONTINUE", True,
                                                           self.selected if self.options == 0 else self.default)
            screen.blit(continue_game_text, continue_game_text.get_rect(center=(GameMetaData.screen_center_width,
                                                                                GameMetaData.screen_center_height - 100)))

        screen.blit(new_game_text, new_game_text.get_rect(center=(GameMetaData.screen_center_width,
                                                                  GameMetaData.screen_center_height - 50)))
        screen.blit(options_text, menu_rect)
        screen.blit(exit_game_text, exit_game_text.get_rect(center=(GameMetaData.screen_center_width,
                                                                    GameMetaData.screen_center_height + 50)))
        pygame.display.update()


class GameScene(SceneBase):
    """Main tetris game scene. When the player is playing."""
    def __init__(self):
        SceneBase.__init__(self)
        
        # This is the column array
        self.empty_line = []
        for i in range(GameMetaData.map_column_no):
            self.empty_line.append(0)
            
        # Insert the column array of zeros into each row to complete the entire map
        self.tetris_map = [self.empty_line[:] for _ in range(GameMetaData.map_row_no)]

        #logger.info("Tetris Map Shape: (%s, %s)", GameMetaData.map_row_no, GameMetaData.map_column_no)

        # Get two shapes. First one being the controlling shape and the second one being the next.
        self.moving_object: List[Shape] = [
            get_random_shape(GameMetaData.map_row_no, GameMetaData.map_column_no),
            get_random_shape(GameMetaData.map_row_no, GameMetaData.map_column_no)
        ]
        
        # Set the rest of the game configurations.
        self.movement_fps = 0
        self.keyboard_speed = 0
        self.movement_speed = 50
        self.maximum_movement_speed = 5
        self.super_speed_mode = False
        self.game_over = False
        self.collision = False

    def process_input(self, events: List[Event]) -> bool:
        """Process key presses in a running game
        
        Args:
            events (List): List of event objects of the key pressed. Also detects mouseOver event.
                           This list can hold more than one item if the user presses the keys quickly
        """
        
        keys = pygame.key.get_pressed()
        
        for event in events:
            
            if event.type == pygame.QUIT:
                return quit_game()
            
            # When the user performs an operation on the falling shape.
            if event.type == pygame.KEYDOWN:
                
                # Pauses when user hits escape key
                if event.key == pygame.K_ESCAPE:
                    # Pauses game. Opens pause menu
                    logger.info("Pausing game...")
                    Scenes.titleScene.is_continue = True
                    Scenes.active_scene = Scenes.titleScene
                
                # Shifts shape to the left
                if event.key == pygame.K_LEFT:
                    self.keyboard_speed = -2
                    self.moving_object[0].move_left(self.tetris_map)
                
                # Shifts shape to the right
                if event.key == pygame.K_RIGHT:
                    self.keyboard_speed = -2
                    self.moving_object[0].move_right(self.tetris_map)
                
                # Pushes shape further down
                if event.key == pygame.K_DOWN and not self.super_speed_mode:
                    self.keyboard_speed = -2
                    self.moving_object[0].move_down(self.tetris_map)
                    logger.debug("Awarded %s Points for Single Push", 2)
                    State.score += 2
                
                # Rotates the orientation of the shape counter-clockwise
                if event.key == pygame.K_UP:
                    could_rotate = self.moving_object[0].rotate(self.tetris_map)
                    if could_rotate:
                        rotate_sound.play()
                    else:
                        tilt_rotate_sound.play()
                
                # Immediately drops shape down to the bottom.
                if event.key == pygame.K_SPACE:
                    
                    # Sets the flag to speed up the shape
                    if not self.super_speed_mode:
                        self.super_speed_mode = True
                        speed_mode_sound.play()
                        self.movement_speed = 1
                    else:
                        self.super_speed_mode = False
                        self.calculate_speed()

        if keys[pygame.K_LEFT]:
            self.keyboard_speed += 1
            if self.keyboard_speed >= 4:
                self.keyboard_speed = 0
                self.moving_object[0].move_left(self.tetris_map)
        
        if keys[pygame.K_RIGHT]:
            self.keyboard_speed += 1
            
            if self.keyboard_speed >= 4:
                self.keyboard_speed = 0
                self.moving_object[0].move_right(self.tetris_map)
                
        return True

    def update(self):
        """
        Counts the FPS ticker so that when it reachs movement_speed
        it moves the shape down one unit then recounts.
        """
        
        self.movement_fps += 1
        if self.movement_fps >= self.movement_speed:
            self.movement_fps = 0
            self.move_object_down_or_game_over()

    def draw_next_shape(self, main_screen):
        self.moving_object[1].draw_next(main_screen, GameMetaData.score_window_text_pos - 20)

    def render(self, screen):
        
        screen.fill(Colour.BLACK.value)

        # Draw Scores
        self.draw_score_area(screen)
        # Draw next shape
        self.draw_next_shape(screen)
        # Draw the moving object to the Screen
        self.moving_object[0].draw(screen)
        # Draw the blocks that are occupied
        self.draw_used_blocks(screen)

        if self.game_over:
            GameScene.draw_game_over()

        pygame.display.update()

    def draw_used_blocks(self, main_screen):
        for row_no, row in enumerate(self.tetris_map):
            for column_no, column_value in enumerate(row):
                if column_value != 0:
                    block_color = get_colour_by_number(column_value)
                    pygame.draw.rect(main_screen, block_color.value, (50 + (column_no * 30), 50 + (row_no * 30), 30, 30), 2)
                    pygame.draw.rect(main_screen, block_color.value, (50 + (column_no * 30) + 5, 50 + (row_no * 30) + 5, 21,
                                                                  21))
        SceneBase.draw_area_grid(main_screen)

    @staticmethod
    def draw_game_over():
        #ogger.info("Drawing Game Over")
        Scenes.titleScene.is_game_over = True
        Scenes.active_scene = Scenes.titleScene

    def calculate_speed(self):
        if State.level < 10:
            new_level = int(State.full_line_no / 10) + 1
            new_movement_speed = 50 - ((new_level - 1) * 5)
                        
            if State.level != new_level:            
                logger.info("Advancing to New Level: %s", new_level)
                logger.info("New Movement Speed: %s", new_movement_speed)
            
            State.level = new_level
            if not self.super_speed_mode and self.movement_speed != new_movement_speed:
                self.movement_speed = new_movement_speed
        elif not self.super_speed_mode:
            self.movement_speed = self.maximum_movement_speed

    def move_object_down_or_game_over(self):
        
        if self.moving_object[0].is_finished_or_collided(self.tetris_map):
            self.movement_speed = 0
            is_game_over = False
            
            # Updating the tetris map by marking the occupied areas.
            for block in self.moving_object[0].blocks:
                row_idx = block[0]
                col_idx = block[1]
                
                # If the shape is at the very top then there is no room game over.
                if row_idx == 0:
                    is_game_over = True

                # Mark this location as 'occupied'
                self.tetris_map[row_idx][col_idx] = get_colour_number_by_name(self.moving_object[0].colour.name)
                self.collision = True
                
            if not is_game_over:
                temp = []
                
                # Full line score
                full_line = 0
                for row in list(reversed(self.tetris_map)):
                    if 0 not in row:
                        full_line += 1
                    else:
                        temp.append(row)
                
                if full_line > 0:
                    
                    logger.debug("Awarded %s Points for a Full Line", full_line * 100)
                    
                    State.full_line_no += full_line
                    State.score += full_line * 100
                    for _ in range(full_line):
                        temp.append(self.empty_line[:])

                self.tetris_map = list(reversed(temp))
                
                self.moving_object.append(get_random_shape(GameMetaData.map_row_no, GameMetaData.map_column_no))
                self.moving_object.pop(0)

            if self.super_speed_mode:
                self.super_speed_mode = False

            self.calculate_speed()

            self.game_over = is_game_over
            #logger.info("Game Over: %s", self.game_over)
            
            #logger.info("Tetris Map:")
            #for row in self.tetris_map:
            #    logger.info("%s", row)

        else:
            # Moves object down a unit
            if self.super_speed_mode:
                State.score += 2
            self.moving_object[0].move_down(self.tetris_map)
    
    def is_block_finished(self) -> bool:
        if self.moving_object[0].is_finished_or_collided(self.tetris_map):
            self.movement_speed = 0
            is_game_over = False
            
            # Updating the tetris map by marking the occupied areas.
            for block in self.moving_object[0].blocks:
                row_idx = block[0]
                col_idx = block[1]
                
                # If the shape is at the very top then there is no room game over.
                if row_idx == 0:
                    return False
                else:
                    return True
        else:
            return False 

def quit_game() -> bool:
    """Quits the game"""
    
    logger.info("Quitting Game...")
    
    # Return false so that while loop stops.
    return False
