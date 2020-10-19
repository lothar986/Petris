from scenes import GameMetaData, TitleScene, Scenes
from shape import *
import os

# Positioned Window
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (100, 100)

#########
# CLOCK #
#########
clock = pg.time.Clock()

main_screen = pg.display.set_mode((GameMetaData.screen_width, GameMetaData.screen_height))
pg.display.set_caption("Petris")

Scenes.titleScene = TitleScene()
Scenes.active_scene = Scenes.titleScene

while True:
    Scenes.active_scene.process_input(pg.event.get())
    Scenes.active_scene.update()
    Scenes.active_scene.render(main_screen)

    clock.tick(50)
