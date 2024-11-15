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


class SequenceDecoder(nn.Module):
    def __init__(self, input_size=306, hidden_size=256, output_size=39, num_layers=3, return_states=False):
        super(SequenceDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_states = return_states

        # self.batch_norm = nn.BatchNorm1d(input_size, eps=1e-3)
        self.batch_norm = nn.BatchNorm1d(input_size, eps=1e-3, momentum=0.01)
        # self.batch_norm = TimeDistributed(nn.BatchNorm1d(input_size, eps=1e-3, momentum=0.01))
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.dense = nn.Linear(self.hidden_size, self.output_size)
        # self.softmax = nn.Softmax(dim=2)
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)  # Xavier Uniform initialization
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)  # Orthogonal initialization is a common choice
            elif 'bias' in name:
                param.data.fill_(0)  # Bias initialized to zero

        init.xavier_uniform_(self.dense.weight)
        if self.dense.bias is not None:
            init.constant_(self.dense.bias, 0.0)

    def forward(self, inputs, initial_state=None):
        """
        :param inputs: (batch_size, seq_len=127, hid_size=306=256+11+39)
        :param initial_state:
                (h, c) h,c: (num_layers=3, batch_size, hid_size=256)
        :return:
            out: (batch_size, seq_len=127, out_size=39)
            states: (h, c) h,c: (num_layers=3, batch_size, hid_size=256)
        """
        # (batch_size, seq_len, hid_size=306)
        x = torch.cat(inputs, dim=-1).float()
        batch_size, seq_len, voc_size = x.size()
        # 这两种batch_norm的方法运算结果接近,但是略有差别
        # x = self.batch_norm(x.float().transpose(1, 2)).transpose(1, 2)
        x = self.batch_norm(x.view(batch_size * seq_len, -1)).view(x.size())
        # x = self.batch_norm(x)
        x, states = self.lstm(x, initial_state)

        out = self.dense(x)
        # out = self.softmax(out)

        if self.return_states:
            return out, states
        else:
            return out
