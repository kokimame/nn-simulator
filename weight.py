import random
import math
import pygame as pg

class Weight:
    eta = 0.75
    alpha = 0.8

    def __init__(self, screen, f_unit, b_unit):
        self.screen = screen
        self.f_unit = f_unit
        self.b_unit = b_unit
        f_unit.b_weights.append(self)
        b_unit.f_weights.append(self)
        self.w = random.uniform(-0.3, 0.3)
        self.prev_dw = 0

    def update(self):
        f_out = self.f_unit.output
        dsig = self.f_unit.d_sigmoid(f_out)

        if(self.f_unit.id == "o"):
            self.delta = (self.f_unit.t_signal - f_out) * dsig
        else:
            self.delta = sum([f_w.delta * f_w.w for f_w in self.f_unit.f_weights]) * dsig

        dw = self.delta * Weight.eta * self.b_unit.output + Weight.alpha * self.prev_dw
        self.w += dw
        self.prev_dw = dw

    def paint(self):
        pg.draw.line(self.screen, pg.Color("black"), self.b_unit.pos, self.f_unit.pos, 2 * self.width())

    def width(self):
>>>>>>> 33db57e62b4792a5f1bdf1b44d4dfa2fe2d1a936
        return int(6 / (1 + math.exp(-2 * self.w)))
