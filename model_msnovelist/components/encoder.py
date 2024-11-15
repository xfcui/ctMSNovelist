from torch import nn
import torch
import numpy as np
from torch.nn import init


class FingerprintFormulaEncoder(nn.Module):
    def __init__(self,
                 layers_hid_size=[512, 256],  # 隐藏层单元
                 layers_decoder=3,
                 units_decoder=256,
                 states_per_layer=2
                 ):
        super(FingerprintFormulaEncoder, self).__init__()
        self.unit1, self.unit2 = layers_hid_size
        self.dec_unit = units_decoder

        self.layers_decoder = layers_decoder  # 3
        self.states_per_layer = states_per_layer  # 2

        # self.batch_norm = nn.BatchNorm1d(3619)
        # self.batch_norm = nn.BatchNorm1d(3619, eps=1e-3)
        self.batch_norm = nn.BatchNorm1d(3619, eps=1e-3, momentum=0.01)

        self.dense1 = nn.Linear(3619, self.unit1)
        self.dense2 = nn.Linear(self.unit1, self.unit2)

        self.rnn_starting_states = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(nn.Linear(self.unit2, self.unit2), nn.ReLU())
                for j in range(states_per_layer)
            ])
            for i in range(layers_decoder)
        ])
        self._initialize_weights()

    def _initialize_weights(self):
        init.xavier_uniform_(self.dense1.weight)
        init.xavier_uniform_(self.dense2.weight)

        if self.dense1.bias is not None:
            init.constant_(self.dense1.bias, 0.0)
        if self.dense2.bias is not None:
            init.constant_(self.dense2.bias, 0.0)

        for states_layer in self.rnn_starting_states:
            for layer in states_layer:
                for sublayer in layer:
                    if isinstance(sublayer, nn.Linear):
                        init.xavier_uniform_(sublayer.weight)
                        if sublayer.bias is not None:
                            init.constant_(sublayer.bias, 0.0)

    def forward(self, inputs):
        inputs = torch.cat(inputs, dim=-1)
        x_bn = self.batch_norm(inputs.float())
        x_bn = self.dense1(x_bn)
        z = self.dense2(x_bn)
        x = z  # (batch_size, 256)
        # ===============================
        # h_states = []
        # c_states = []
        # for state_layers in self.rnn_starting_states:
        #     state_ = []
        #     for state_layer in state_layers:
        #         x = state_layer(x)  # (batch_size, 256)
        #         state_.append(x)
        #     h, c = state_
        #     h_states.append(h)
        #     c_states.append(c)
        #
        # h_states = torch.stack(h_states)  # (3, batch_size, 256)
        # c_states = torch.stack(c_states)  # (3, batch_size, 256)
        # rnn_states = (h_states, c_states)
        # ================================
        rnn_states = [
            [state_layer(x) for state_layer in state_layers]
            for state_layers in self.rnn_starting_states
        ]
        h_states = [i[0] for i in rnn_states]
        c_states = [i[1] for i in rnn_states]
        h_states = torch.stack(h_states)  # (3, batch_size, 256)
        c_states = torch.stack(c_states)  # (3, batch_size, 256)

        rnn_states = (h_states, c_states)
        return z, rnn_states
