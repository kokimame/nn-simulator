import os
import pygame as pg

from network import *

def main():
    pg.init()
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%i,%i" % (600, 200)
    os.environ['SDL_VIDEO_CENTERED'] = '0'
    screen = pg.display.set_mode((600, 500))
    done = False
    learned = False
    clock = pg.time.Clock()

    nn = Network(screen)

    while not done:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                done = True
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                done = True

        done = nn.is_training_done()

        screen.fill(pg.Color("white"))
        nn.paint()

        pg.display.flip()
        clock.tick(60)

    # nn.training_phase()

if __name__ == "__main__":
    main()
