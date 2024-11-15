import torch
from torch import nn

from ..components.Hint import myRNN
from ..components.recurrent_additive import RecurrentAdditiveCell


class GlobalSumPooling(nn.Module):
    def __init__(self):
        super(GlobalSumPooling, self).__init__()
        self.step_counter_factor = lambda x: torch.ones_like(x)
        self.step_counter = myRNN(RecurrentAdditiveCell(units=1, factor=1),
                                  return_sequences=False,
                                  return_state=False)
        self.average_pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, inputs):
        step_counter_factor = self.step_counter_factor(inputs)  # (batch_size,seq_len,1)
        # batch_size, seq_len, _ = step_counter_factor.size()
        step_counter = self.step_counter(step_counter_factor)  # (batch_size,1)

        average = self.average_pooling(inputs.permute(0, 2, 1)).squeeze(-1)

        global_sum = step_counter * average
        return global_sum  # (batch_size, 1)
