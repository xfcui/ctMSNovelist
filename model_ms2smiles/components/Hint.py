import torch
from torch import nn
import numpy as np

from ..components.recurrent_additive import RecurrentAdditiveCell


class myRNN(nn.Module):
    def __init__(self, rnn_cell=RecurrentAdditiveCell(units=11, factor=-1),
                 return_sequences=False, return_state=False):
        super(myRNN, self).__init__()
        self.rnn_cell = rnn_cell
        self.return_sequences = return_sequences
        self.return_state = return_state

    def forward(self, x, initial_state=None):
        # initial_state==None: GlobalSumPooling (batch_size,seq_len,hid_size)-->(batch_size,hid_size)
        # initial_state!=None: auxiliary_counter (batch_size,seq_len,hid_size)-->(batch_size,seq_len,hid_size)
        # x: (batch_size,seq_len, 11)  (512,128,11)
        # state: (batch_size, 11)  (512,11)
        step_counter = initial_state
        batch_size = x.size(0)
        seq_len = x.size(1)
        if initial_state is None:
            step_counter = torch.zeros(batch_size, 1).cuda()
        y_t = []
        for i in range(seq_len):
            # (batch_size, 11) # [(batch_size, 11)]
            step_counter, state = self.rnn_cell(x[:, i, :], [step_counter])
            y_t.append(step_counter)
        y_t = torch.stack(y_t).transpose(0, 1)  # (batch_size, seq_len, 1)

        # out = None
        if self.return_sequences:  # auxiliary_counter
            out = y_t  # (batch_size,seq_len,hid_size)
        else:  # GlobalSumPooling
            out = step_counter  # (batch_size,hid_size)
        if self.return_state:
            out = [out, state[0]]

        return out
