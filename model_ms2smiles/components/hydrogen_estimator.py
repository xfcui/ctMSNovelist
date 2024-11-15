import torch
from torch import nn
from torch.nn import init


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        # 输入形状: (batch_size, time_steps, input_size)
        batch_size, time_steps, input_size = x.size()

        # 重塑输入: (batch_size * time_steps, input_size)
        x_reshaped = x.reshape(-1, input_size)

        # 应用模块 (如Dense层)
        y = self.module(x_reshaped)

        # 重塑回原始形状: (batch_size, time_steps, output_size)
        output_size = y.size(-1)
        y = y.reshape(batch_size, time_steps, output_size)
        return y

# class TimeDistributed(nn.Module):
#     def __init__(self, module):
#         super(TimeDistributed, self).__init__()
#         self.module = module
#
#     def forward(self, x):
#         outputs = []
#         seq_len = x.size(1)
#         for t in range(seq_len):
#             xt = x[:, t, :]
#             output = self.module(xt)
#             outputs.append(output)
#         outputs = torch.stack(outputs, dim=1)
#         return outputs


class HydrogenEstimator(nn.Module):
    def __init__(self, layers=2, units=32, pad_term_mask=None, return_states=False):
        super(HydrogenEstimator, self).__init__()
        self.layers_ = layers
        self.units_ = units
        self.pad_term_mask = pad_term_mask.cuda()
        self.return_states = return_states

        self.lstm = nn.LSTM(input_size=39, hidden_size=self.units_, num_layers=self.layers_, batch_first=True)
        # TimeDistributed(nn.Linear())与nn.Linear()效果相同
        self.out_layer = nn.Linear(self.units_, 1)

        # self.dense = nn.Linear(self.units_, 1)
        # self.out_layer = TimeDistributed(self.dense)

        # self.out_layer = TimeDistributed(nn.Linear(self.units_, 1))
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)  # Xavier Uniform initialization
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)  # Orthogonal initialization is a common choice
            elif 'bias' in name:
                param.data.fill_(0)  # Bias initialized to zero

        init.xavier_uniform_(self.out_layer.weight)
        if self.out_layer.bias is not None:
            init.constant_(self.out_layer.bias, 0.0)

    def forward(self, inputs, initial_state=None):
        """
        :param inputs: (batch_size, seq_len, voc_size)  (256,127,39)
        :param initial_state:
                1-test、(h, c) h,c: (num_layers=2, batch_size, hid_size=32)
                2-train、None
        :return:
            layer_stack: (batch_size, seq_len=127, out_size=1)
            states: (h, c) h,c: (num_layers=2, batch_size, hid_size=32)
        """
        # (batch_size, seq_len, voc_size)  (256,127,39)
        tokens_input = inputs

        layer_stack, states = self.lstm(tokens_input, initial_state)

        # (batch_size, seq_len, out_size)  (256,127,1)
        layer_stack = self.out_layer(layer_stack)
        # layer_stack = self.relu(layer_stack)
        # If applicable, mask pad and termination character outputs
        if self.pad_term_mask is not None:
            # (batch_size, seq_len, out_size)
            mask_vec = torch.max(tokens_input * self.pad_term_mask, dim=2, keepdim=True)[0]
            layer_stack = layer_stack * mask_vec

        if self.return_states:
            return layer_stack, states
        else:
            return layer_stack  # (batch_size, seq_len, out_size)
