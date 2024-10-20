from torch import nn
import torch
import numpy as np
from torch.nn import init



class FingerprintFormulaEncoder(nn.Module):
    def __init__(self,
                 unit1=512,
                 unit2=256,
                 decoder_layers=3,
                 states_per_layer=2,
                 fp_dim=3609,
                 mf_dim=10
                 ):
        super(FingerprintFormulaEncoder, self).__init__()
        self.unit1 = unit1
        self.unit2 = unit2
        self.decoder_layers = decoder_layers
        self.states_per_layer = states_per_layer

        self.dense1 = nn.Linear(fp_dim, self.unit1)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(self.unit1, self.unit2)
        self.relu2 = nn.ReLU()

        self.dense3 = nn.Linear(mf_dim, self.unit2)
        self.relu3 = nn.ReLU()
        self.dense4 = nn.Linear(self.unit2, self.unit2)
        self.relu4 = nn.ReLU()
        self.dense5 = nn.Linear(self.unit1, self.unit2)
        self.relu5 = nn.ReLU()

        self.dense6 = nn.Linear(self.unit2, self.unit2)
        self.layer_norm1 = nn.LayerNorm(self.unit2)
        # self.batch_norm1 = nn.BatchNorm1d(self.unit2)
        self.dense7 = nn.Linear(self.unit2, self.unit2)
        self.layer_norm2 = nn.LayerNorm(self.unit2)
        # self.batch_norm2 = nn.BatchNorm1d(self.unit2)

        self.dense1 = nn.Linear(fp_dim, self.unit1)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(self.unit1, self.unit2)
        self.relu2 = nn.ReLU()

        self.fc_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(self.unit2, self.unit2),
            nn.ReLU(),
            nn.Linear(self.unit2, self.unit2),
            nn.ReLU()
        ) for _ in range(3)])
        self.alpha = nn.Parameter(torch.ones(self.unit2), requires_grad=True)

    def forward(self, encode_fp, encode_mf):
        encode_fp = self.relu1(self.dense1(encode_fp.float()))
        encode_fp = self.relu2(self.dense2(encode_fp))
        encode_fp = self.layer_norm1(self.dense6(encode_fp))
        # encode_fp = self.batch_norm1(self.dense6(encode_fp))

        encode_mf = self.relu3(self.dense3(encode_mf.float()))
        encode_mf = self.relu4(self.dense4(encode_mf))
        encode_mf = self.layer_norm2(self.dense7(encode_mf))
        # encode_mf = self.batch_norm2(self.dense7(encode_mf))

        emb = torch.cat([encode_fp, self.alpha * encode_mf], dim=1)
        z = self.relu5(self.dense5(emb))
        x = z
        state = []
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            state.append(x)
        state_stack = torch.stack(state, dim=0)
        return z, state_stack
