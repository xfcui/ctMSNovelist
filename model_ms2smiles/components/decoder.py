import torch
from torch import nn
from torch.nn import init

class SequenceDecoder(nn.Module):
    def __init__(self, input_size=592, hidden_size=256, output_size=39, num_layers=3, return_states=False):
        # input_size: 592 or 603
        super(SequenceDecoder, self).__init__()
        self.return_states = return_states
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size if i == 0 else hidden_size, hidden_size, batch_first=True) for i in
            range(self.num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(self.num_layers)
        ])
        self.dense = nn.Linear(self.hidden_size, output_size)

    def forward(self, x, initial_state=None):
        h_stack = []
        c_stack = []
        for i in range(self.num_layers):
            lstm_layer = self.lstm_layers[i]
            ln_layer = self.layer_norms[i]
            hx, cx = initial_state
            lstm_output, (h_n, c_n) = lstm_layer(x, (hx[i].unsqueeze(0), cx[i].unsqueeze(0)))
            x = ln_layer(lstm_output)

            h_stack.append(h_n)
            c_stack.append(c_n)

        x = self.dense(x)
        if self.return_states:
            h_stack = torch.cat(h_stack, dim=0)
            c_stack = torch.cat(c_stack, dim=0)
            return x, (h_stack, c_stack)
        return x
