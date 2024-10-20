import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================================================================
class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attention_weights = self.softmax(scores)
        attended = torch.matmul(attention_weights, V)
        return attended, attention_weights


class SMILESToFPModelRNN(nn.Module):
    def __init__(self, input_size=39, hidden_size=128, output_size=3609, num_layers=2, dropout=0.1):
        super(SMILESToFPModelRNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = SelfAttention(hidden_size * 2, hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        attended, _ = self.attention(rnn_out)
        attended_flat = attended.mean(dim=1)
        x = self.fc1(attended_flat)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = self.sigmoid(x)
        return output

# ===================================================================================
class EncoderVAE(nn.Module):
    def __init__(self,
                 smiles_input_size,
                 struct_input_size,
                 hidden_size,
                 latent_dim,
                 num_layers=2):
        super(EncoderVAE, self).__init__()
        self.lstm = nn.LSTM(smiles_input_size, hidden_size, num_layers, batch_first=True)
        self.fc_struct = nn.Linear(struct_input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size * 2, latent_dim)

    def forward(self, smiles, struct_fp):
        _, (h, _) = self.lstm(smiles)
        h = h[-1]  # 使用最后一层的隐藏状态
        struct_h = torch.relu(self.fc_struct(struct_fp))  # 提取struct_fp特征
        combined_h = torch.cat((h, struct_h), dim=-1)  # 将LSTM输出和struct_fp特征拼接
        mu = self.fc_mu(combined_h)
        logvar = self.fc_logvar(combined_h)
        return mu, logvar
class Reparameterization(nn.Module):
    def __init__(self):
        super(Reparameterization, self).__init__()

    def forward(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class DecoderVAE(nn.Module):
    def __init__(self,
                 latent_dim,
                 struct_input_size,
                 hidden_size,
                 output_size,
                 num_layers=2):
        super(DecoderVAE, self).__init__()
        self.fc1 = nn.Linear(latent_dim + struct_input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, struct_fp):
        z_struct_combined = torch.cat((z, struct_fp), dim=-1)  # 将潜在向量和struct_fp拼接
        z_struct_combined = self.fc1(z_struct_combined).unsqueeze(1)  # (batch_size, 1, hidden_size)
        _, (h, _) = self.lstm(z_struct_combined)
        h = h[-1]
        output = self.sigmoid(self.fc2(h))
        return output


class VAE(nn.Module):
    def __init__(self,
                 smiles_input_size=39,
                 struct_input_size=3609,
                 hidden_size=256,
                 latent_dim=128,
                 output_size=3609,
                 num_layers=2):
        super(VAE, self).__init__()
        self.encoder = EncoderVAE(smiles_input_size, struct_input_size, hidden_size, latent_dim, num_layers)
        self.reparameterization = Reparameterization()
        self.decoder = DecoderVAE(latent_dim, struct_input_size, hidden_size, output_size, num_layers)

    def forward(self, smiles, struct_fp):
        mu, logvar = self.encoder(smiles, struct_fp)
        z = self.reparameterization(mu, logvar)
        reconstructed_x = self.decoder(z, struct_fp)
        return reconstructed_x, mu, logvar
# ===================================================================================
class SMILESToFPGenerator(nn.Module):
    def __init__(self, input_size=39, hidden_size=256, output_size=3609, num_layers=2, dropout=0.1):
        super(SMILESToFPGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # self.attention = SelfAttention(hidden_size * 2, hidden_size * 2)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(output_size * 2, output_size)
        )
        self.sigmoid = nn.Sigmoid()
        self.final_jitter = 0.1

    def forward(self, x):
        bs = x.size(0)
        rnn_out, _ = self.rnn(x)
        # attended, _ = self.attention(rnn_out)
        rnn_flat = rnn_out.mean(dim=1)
        fc_out = self.fc_layers(rnn_flat)

        noise = torch.rand((bs, self.hidden_size)).to(x.device)
        noise_out = self.fc_layers(noise)

        cat = torch.cat([fc_out, noise_out], dim=-1)
        cat_out = self.fc(cat)

        sig_out = self.sigmoid(cat_out)
        # fp_noise = self.final_jitter * (torch.rand_like(sig_out) - 0.5)
        # out = torch.clamp(sig_out + fp_noise,
        #                   sig_out.min(),
        #                   sig_out.max())
        return sig_out
# ===================================================================================
class FPToFPModel(nn.Module):
    def __init__(self, input_size=3609, hidden_size=1024):
        super(FPToFPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# ===================================================================================
class SMILESToSIModel(nn.Module):
    def __init__(self):
        super(SMILESToSIModel, self).__init__()
        self.SMILESToSTRUCT = SMILESToFPModelRNN()
        self.STRUCT2SIM = FPToFPModel()

    def forward(self, gt):
        struct_fp = self.SMILESToSTRUCT(gt)
        struct_fp_ = struct_fp.detach().round()
        sim_fp = self.STRUCT2SIM(struct_fp_)
        return struct_fp, sim_fp
# ===================================================================================
class TransformerModel(nn.Module):
    def __init__(self,
                 vocab_size=39,
                 d_model=512,
                 nhead=8,
                 num_layers=3,
                 output_size=3609,
                 max_seq_len=127):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = self.create_position_encoding(max_seq_len, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, output_size)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def create_position_encoding(self, max_seq_len, d_model):
        # 生成位置编码
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pos_encoding = torch.zeros(max_seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)  # (1, max_seq_len, d_model)
        return pos_encoding

    def _init_weights(self):
        # 初始化权重
        nn.init.kaiming_normal_(self.embedding.weight, mode='fan_out', nonlinearity='relu')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x的形状为 (batchsize, seqlen)
        x = self.embedding(x)  # (batchsize, seqlen, d_model)
        seq_len = x.size(1)
        pos_encoding = self.position_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_encoding  # 加上位置编码
        x = x.transpose(0, 1)  # Transformer需要 (seqlen, batchsize, d_model)
        x = self.transformer(x)  # (seqlen, batchsize, d_model)
        x = x.mean(dim=0)  # (batchsize, d_model)
        x = self.fc(x)  # (batchsize, output_size)
        x = self.sigmoid(x)
        return x


# ===================================================================================
class Generator(nn.Module):
    def __init__(self, input_size=3609, hid_size=1024, output_size=3609):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(2048, output_size),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, noise):
        x = self.model(x)
        noise = self.model(noise)
        x = torch.cat([x, noise], dim=-1)
        x = self.fc(x)
        out = self.sigmoid(x)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_size=3609):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x
# ===================================================================================
# class SMILESToFPModelRNN(nn.Module):
#     def __init__(self,
#                  input_size=39,
#                  num_classes=3609,
#                  num_heads=8,
#                  num_layers=6,
#                  seq_len=127,
#                  hidden_dim=512,
#                  dropout=0.1):
#         super(SMILESToFPModelRNN, self).__init__()
#
#         # Positional encoding
#         self.positional_encoding = PositionalEncoding(input_size, dropout)
#
#         # Transformer encoder
#         self.transformer_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=input_size,
#                 nhead=num_heads,
#                 dim_feedforward=hidden_dim,
#                 dropout=dropout
#             ),
#             num_layers=num_layers
#         )
#
#         # Fully connected layer for final classification
#         self.fc = nn.Linear(input_size * seq_len, num_classes)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # Apply positional encoding
#         x = self.positional_encoding(x)
#
#         # Pass through transformer encoder
#         x = self.transformer_encoder(x)
#
#         # Flatten the output
#         x = x.view(x.size(0), -1)
#
#         # Pass through fully connected layer
#         out = self.fc(x)
#         out = self.sigmoid(out)
#         return out
#
#
# class PositionalEncoding(nn.Module):
#     """Positional encoding."""
#
#     def __init__(self, num_hiddens, dropout, max_len=1000):
#         super().__init__()
#         self.dropout = nn.Dropout(dropout)
#         # Create a long enough P
#         self.P = torch.zeros((1, max_len, num_hiddens))
#         X = torch.arange(max_len, dtype=torch.float32).reshape(
#             -1, 1) / torch.pow(10000, torch.arange(
#             0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
#         self.P[:, :, 0::2] = torch.sin(X)
#         self.P[:, :, 1::2] = torch.cos(X)
#
#     def forward(self, X):
#         X = X + self.P[:, :X.shape[1], :].to(X.device)
#         return self.dropout(X)


# ===================================================================================
# class SMILESToFPModelRNN(nn.Module):
#     def __init__(self,
#                  input_size=39,
#                  lstm_hidden_size=256,
#                  num_layers=2,
#                  # seq_len=127,
#                  dense_hidden_size=4096,
#                  out_size=3609
#                  ):
#         super(SMILESToFPModelRNN, self).__init__()
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden_size, num_layers=num_layers,
#                             batch_first=True)
#         self.dropout = nn.Dropout(p=0.5)
#
#         self.fc1 = nn.Linear(lstm_hidden_size, dense_hidden_size)
#         self.layer_norm1 = nn.LayerNorm(dense_hidden_size)
#         self.relu = nn.ReLU()
#
#         self.fc1 = nn.Linear(lstm_hidden_size, dense_hidden_size)
#         self.layer_norm1 = nn.LayerNorm(dense_hidden_size)
#         self.relu = nn.ReLU()
#
#         self.fc2 = nn.Linear(dense_hidden_size, dense_hidden_size)
#         self.layer_norm2 = nn.LayerNorm(dense_hidden_size)
#
#         self.fc3 = nn.Linear(dense_hidden_size, out_size)
#         self.layer_norm3 = nn.LayerNorm(out_size)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, inputs):
#         lstm_out, _ = self.lstm(inputs)
#         # lstm_out = self.dropout(lstm_out)
#
#         fc1_out = self.fc1(lstm_out[:, -1, :])
#         ln1_out = self.layer_norm1(fc1_out)
#         relu_out = self.relu(ln1_out)
#
#         fc2_out = self.fc2(relu_out)
#         ln2_out = self.layer_norm2(fc2_out)
#         relu2_out = self.relu(ln2_out) + relu_out
#
#         fc3_out = self.fc3(relu2_out)
#         ln3_out = self.layer_norm3(fc3_out)
#         out = self.sigmoid(ln3_out)
#         return out
# class SMILESToFPModelRNN(nn.Module):
#     def __init__(self,
#                  input_size=39,
#                  lstm_hidden_size=256,
#                  num_layers=2,
#                  seq_len=127,
#                  dense_hidden_size=4096,
#                  out_size=3609
#                  ):
#         super(SMILESToFPModelRNN, self).__init__()
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden_size, num_layers=num_layers, batch_first=True)
#         self.dropout = nn.Dropout(p=0.5)
#
#         self.fc1 = nn.Linear(lstm_hidden_size * seq_len, dense_hidden_size)
#         self.layer_norm1 = nn.LayerNorm(dense_hidden_size)
#         self.relu = nn.ReLU()
#
#         self.fc2 = nn.Linear(dense_hidden_size, out_size)
#         self.layer_norm2 = nn.LayerNorm(out_size)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, inputs):
#         lstm_out, _ = self.lstm(inputs)
#         lstm_out = lstm_out.contiguous().view(lstm_out.size(0), -1)
#         lstm_out = self.dropout(lstm_out)
#
#         fc1_out = self.fc1(lstm_out)
#         ln1_out = self.layer_norm1(fc1_out)
#         relu_out = self.relu(ln1_out)
#
#         fc2_out = self.fc2(relu_out)
#         ln2_out = self.layer_norm2(fc2_out)
#         out = self.sigmoid(ln2_out)
#         return out
