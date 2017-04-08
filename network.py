from units import *
from weight import *

class Network:
    err_bound = 0.08
    offset = (80, 120) # Offset of screen

    def __init__(self, screen):
        self.screen = screen
        self.weights = []
        self.demo_setting()
        self.create_units()
        self.connect_weights()
        self.debug_w_setting()

    def propagation(self):
        [hid_u.output_update() for hid_u in self.hid_layer]
        [out_u.output_update() for out_u in self.out_layer]

    def weight_update(self):
        for weights in [unit.b_weights for unit in self.out_layer]:
            [w.update() for w in weights]
        for weights in [unit.b_weights for unit in self.hid_layer]:
            [w.update() for w in weights]

    def is_training_done(self):
        self.sum_err = 0
        for i in range(self.n_pat):
            self.set_pattern(i)
            self.propagation()
            self.weight_update()
            self.debug_learning()
            self.sum_err += self.compute_error()
        print(">>> Error: %f" % (self.sum_err))

        return (self.sum_err < Network.err_bound)

    def training_phase(self):
        count = 0
        while self.sum_err > Network.err_bound:
            self.sum_err = 0
            count += 1
            for i in range(self.n_pat):
                self.set_pattern(i)
                self.propagation()
                self.weight_update()
                self.debug_learning()
                self.sum_err += self.compute_error()
            print("                 Error: %f(%d)" % (self.sum_err, count))

    def compute_error(self):
        return sum([(unit.t_signal - unit.output)**2 for unit in self.out_layer]) * 0.5

    def set_pattern(self, n):
        dict_pat = {0:[1,1,0], 1:[1,0,1], 2:[0,1,1], 3:[0,0,0]}
        pattern = dict_pat[n]
        self.in_layer[0].output = pattern[0]
        self.in_layer[1].output = pattern[1]
        self.out_layer[0].t_signal = pattern[2]

    def paint(self):
        for weights in self.weights:
            [w.paint() for w in weights]
        [unit.paint() for unit in self.in_layer]
        [unit.paint() for unit in self.hid_layer]
        [unit.paint() for unit in self.out_layer]

    def demo_setting(self):
        self.n_in = 2
        self.n_hid = 4
        self.n_out = 1
        self.n_pat = 4
        self.sum_err = 1

    def manual_setting(self):
        # Setup the number of units through console input
        pass

    def create_units(self):
        self.in_layer = [InputUnit(self.screen,(Network.offset[0], self.units_y(i, self.n_in)), i)
                            for i in range(self.n_in)]
        self.hid_layer = [HiddenUnit(self.screen, (int(self.screen.get_size()[0] / 2),
                            self.units_y(i, self.n_hid)), i) for i in range(self.n_hid)]
        self.out_layer = [OutputUnit(self.screen, (self.screen.get_size()[0] - Network.offset[0],
                            self.units_y(i, self.n_out)), i) for i in range(self.n_out)]

    def units_y(self, stop, n):
        # Currently only considering the case of 3 layer (input, hidden and output)
        scr_y = self.screen.get_size()[1]
        if n == 1:
            return int(scr_y / 2)
        else:
            height = (scr_y - 2 * Network.offset[1]) / (n - 1)
            return Network.offset[1] + int(height * stop)

    def connect_weights(self):
        self.weights.append(
            [Weight(self.screen, hid_u, in_u) for hid_u in self.hid_layer for in_u in self.in_layer])
        self.weights.append(
            [Weight(self.screen, out_u, hid_u) for out_u in self.out_layer for hid_u in self.hid_layer])

    def debug_learning(self):
        print("Input: (%d, %d)/ Target: %d / Output: %f" %
              (self.in_layer[0].output, self.in_layer[1].output, self.out_layer[0].t_signal, self.out_layer[0].output))

    def debug_output(self):
        for layer in self.layout:
            for unit in layer:
                unit.print_mark()
            print()

    def debug_connection(self):
        for unit in self.in_layer:
            for w in unit.f_weights:
                print("%s%d is connected to %s%d" % (unit.id, unit.num, w.f_unit.id, w.f_unit.num))
        for unit in self.hid_layer:
            for w in unit.f_weights:
                print("%s%d is connected to %s%d" % (unit.id, unit.num, w.f_unit.id, w.f_unit.num))
            for w in unit.b_weights:
                print("%s%d is connected to %s%d" % (unit.id, unit.num, w.b_unit.id, w.b_unit.num))
        for unit in self.out_layer:
            for w in unit.b_weights:
                print("%s%d is connected to %s%d" % (unit.id, unit.num, w.b_unit.id, w.b_unit.num))

    def debug_w_setting(self):
        print("L:HID: ", end="")
        for unit in self.hid_layer:
            print([weight.w for weight in unit.b_weights], end="")
        print("\nL:OUT: ", end="")
        for unit in self.out_layer:
            print([weight.w for weight in unit.b_weights], end="")
        print()
