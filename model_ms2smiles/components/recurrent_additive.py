from torch import nn


class RecurrentAdditiveCell(nn.Module):
    def __init__(self, units, factor):
        super(RecurrentAdditiveCell, self).__init__()

        self.units = units
        self.state_size = units
        self.factor = factor

    def forward(self, inputs, states):
        prev_output = states[0]
        output = prev_output + self.factor * inputs

        return output, [output]  # output, state
