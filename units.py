import math
import pygame as pg

class Unit:
    radius = 15

    def __init__(self, screen, pos, id, num):
        self.screen = screen
        self.pos = pos
        self.id = id
        self.num = num

    def print_mark(self):
        print(" %s%d " % (self.id, self.num), end="")

    def sigmoid(self, x):
        return  1.0 / (1.0 + math.exp(-x))

    def d_sigmoid(self, x):
        return x * (1.0 - x)

    def paint(self):
        pg.draw.circle(self.screen, pg.Color('red'), self.pos, Unit.radius)


class InputUnit(Unit):
    def __init__(self, screen, pos, num):
        Unit.__init__(self, screen, pos, "i", num)
        self.f_weights = []
        self.output = 0


class HiddenUnit(Unit):
    def __init__(self, screen, pos, num):
        Unit.__init__(self, screen, pos, "h", num)
        self.f_weights = []
        self.b_weights = []

    def output_update(self):
        self.output = self.sigmoid(sum([b_w.b_unit.output * b_w.w for b_w in self.b_weights]))
        #print("%s%d: output=%f" % (self.id, self.num, self.output))


class OutputUnit(Unit):
    def __init__(self, screen, pos, num):
        Unit.__init__(self, screen, pos, "o", num)
        self.b_weights = []
        self.t_signal = 0

    def output_update(self):
        self.output = self.sigmoid(sum([b_w.b_unit.output * b_w.w for b_w in self.b_weights]))
        #print("%s%d: output=%f" % (self.id, self.num, self.output))
