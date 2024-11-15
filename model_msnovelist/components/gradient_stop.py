from torch import nn


class GradientStop(nn.Module):
    def __init__(self):
        super(GradientStop, self).__init__()

    def forward(self, x):
        return x.detach()
