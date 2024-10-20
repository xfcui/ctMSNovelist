from torch import nn
import torch

from .transcoder_model import TranscoderModel


class TransVAEModel(nn.Module):
    def __init__(self):
        super(TransVAEModel, self).__init__()
        # self.vae = VAE()
        self.vae = VAE_V2()
        # self.vae = VAE_V3
        self.transcoder = TranscoderModel()

    def forward(self, inputs):
        gt = inputs["tokens_X"]
        sim_fp, mu, logvar = self.vae(gt)

        transcoder_input = {"FP": sim_fp.detach(),
                            "MF": inputs['MF'],
                            "tokens_X": inputs["tokens_X"]}
        dec_out, estimated_h_sum = self.transcoder(transcoder_input)

        return sim_fp, mu, logvar, dec_out, estimated_h_sum

# ================================V0=======================================
# class EncoderVAE(nn.Module):
#     def __init__(self,
#                  smiles_input_size,
#                  hidden_size,
#                  latent_dim,
#                  num_layers=2):
#         super(EncoderVAE, self).__init__()
#         self.lstm = nn.LSTM(smiles_input_size, hidden_size, num_layers, batch_first=True)
#         self.fc_mu = nn.Linear(hidden_size, latent_dim)
#         self.fc_logvar = nn.Linear(hidden_size, latent_dim)
#
#     def forward(self, smiles):
#         _, (h, _) = self.lstm(smiles)
#         h = h[-1]  # 使用最后一层的隐藏状态
#         mu = self.fc_mu(h)
#         logvar = self.fc_logvar(h)
#         return mu, logvar
#
# class Reparameterization(nn.Module):
#     def __init__(self):
#         super(Reparameterization, self).__init__()
#
#     def forward(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
# class DecoderVAE(nn.Module):
#     def __init__(self,
#                  latent_dim=128,
#                  hidden_size=256,
#                  output_size=3609
#                  ):
#         super(DecoderVAE, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(latent_dim, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, output_size)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, z):
#         fc_out = self.fc(z)
#         output = self.sigmoid(fc_out)
#         return output
#
#
# class VAE(nn.Module):
#     def __init__(self,
#                  smiles_input_size=39,
#                  hidden_size=256,
#                  latent_dim=256, # 128
#                  output_size=3609,
#                  num_layers=2):
#         super(VAE, self).__init__()
#         self.encoder = EncoderVAE(smiles_input_size, hidden_size, latent_dim, num_layers)
#         self.reparameterization = Reparameterization()
#         self.decoder = DecoderVAE(latent_dim, hidden_size, output_size)
#
#     def forward(self, smiles):
#         mu, logvar = self.encoder(smiles)
#         z = self.reparameterization(mu, logvar)
#         reconstructed_x = self.decoder(z)
#         return reconstructed_x, mu, logvar


# ================================ V1 =======================================
# 这个版本可能能在完整的GNPS上复现或是超越msnovelist
# 这个模型在fold4-gnps上，在valid_smiles, correct_MF上可以超越msnovelist, retrieval比msnovelist低不超过2%

class Reparameterization(nn.Module):
    def __init__(self):
        super(Reparameterization, self).__init__()

    def forward(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std



class VAE(nn.Module):
    def __init__(self,
                 smiles_input_size=39,
                 hidden_size=256,
                 latent_dim=256,
                 output_size=3609,
                 num_layers=2):
        super(VAE, self).__init__()
        self.encoder = self.EncoderVAE(smiles_input_size, hidden_size, latent_dim, num_layers)
        self.reparameterization = Reparameterization()
        self.decoder = self.DecoderVAE(latent_dim, hidden_size, output_size)

    class EncoderVAE(nn.Module):
        def __init__(self, smiles_input_size, hidden_size, latent_dim, num_layers=2):
            super(VAE.EncoderVAE, self).__init__()
            self.lstm = nn.LSTM(smiles_input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
            self.fc_mu = nn.Linear(hidden_size * 2, latent_dim)  # 双向LSTM的输出需要乘以2
            self.fc_logvar = nn.Linear(hidden_size * 2, latent_dim)

        def forward(self, smiles):
            _, (h, _) = self.lstm(smiles)
            h = torch.cat((h[-2], h[-1]), dim=1)  # 拼接双向LSTM的最后一层输出
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar

    class DecoderVAE(nn.Module):
        def __init__(self, latent_dim=256, hidden_size=256, output_size=3609):
            super(VAE.DecoderVAE, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(latent_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
                nn.Sigmoid()
            )

        def forward(self, z):
            return self.fc(z)

    def forward(self, smiles):
        mu, logvar = self.encoder(smiles)
        z = self.reparameterization(mu, logvar)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, logvar

# ================================ V2 (t18)=======================================
class VAE_V2(nn.Module):
    def __init__(self,
                 smiles_input_size=39,
                 lstm_hidden_size=256,
                 hidden_size=512,
                 latent_dim=512,
                 output_size=3609,
                 num_layers=2):
        super(VAE_V2, self).__init__()
        self.encoder = self.EncoderVAE(smiles_input_size, lstm_hidden_size, latent_dim, num_layers)
        self.reparameterization = Reparameterization()
        self.decoder = self.DecoderVAE(latent_dim, hidden_size, output_size)

    class EncoderVAE(nn.Module):
        def __init__(self,
                     smiles_input_size=39,
                     hidden_size=256,
                     latent_dim=512,
                     num_layers=2):
            super(VAE_V2.EncoderVAE, self).__init__()
            self.lstm = nn.LSTM(smiles_input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
            self.fc1 = nn.Sequential(
                nn.Linear(hidden_size * 2, latent_dim),
                nn.ReLU()
            )
            self.fc_mu = nn.Linear(latent_dim, latent_dim)  # 双向LSTM的输出需要乘以2
            self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        def forward(self, smiles):
            _, (h, _) = self.lstm(smiles)
            h = torch.cat((h[-2], h[-1]), dim=1)  # 拼接双向LSTM的最后一层输出
            residual = h
            h = self.fc1(h)
            h = h + residual

            residual = h
            mu = self.fc_mu(h)
            mu = mu + residual
            logvar = self.fc_logvar(h)
            logvar = logvar + residual
            return mu, logvar

    class DecoderVAE(nn.Module):
        def __init__(self,
                     latent_dim=512,
                     hidden_size=512,
                     output_size=3609):
            super(VAE_V2.DecoderVAE, self).__init__()
            self.fc1 = nn.Linear(latent_dim, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size)
            self.fc4 = nn.Linear(hidden_size, output_size)
            self.sigmoid = nn.Sigmoid()
            self.relu = nn.ReLU()

        def forward(self, z):
            # 第一层和第二层之间的残差连接
            residual = z
            h = self.relu(self.fc1(z))
            h = h + residual

            # 第二层和第三层之间的残差连接
            residual = h
            h = self.relu(self.fc2(h))
            h = h + residual

            # 第三层和第四层之间的残差连接
            residual = h
            h = self.relu(self.fc3(h))
            h = h + residual

            # 最后一层没有残差连接
            output = self.fc4(h)
            return self.sigmoid(output)

    def forward(self, smiles):
        mu, logvar = self.encoder(smiles)
        z = self.reparameterization(mu, logvar)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, logvar

# ================================ V3 =======================================
class VAE_V3(nn.Module):
    def __init__(self,
                 smiles_input_size=39,
                 lstm_hidden_size=256,
                 hidden_size=512,
                 latent_dim=512,
                 output_size=3609,
                 num_layers=2):
        super(VAE_V3, self).__init__()
        self.encoder = self.EncoderVAE(smiles_input_size, lstm_hidden_size, latent_dim, num_layers)
        self.reparameterization = Reparameterization()
        self.decoder = self.DecoderVAE(latent_dim, hidden_size, output_size)

    class EncoderVAE(nn.Module):
        def __init__(self,
                     smiles_input_size=39,
                     hidden_size=256,
                     latent_dim=512,
                     num_layers=2):
            super(VAE_V3.EncoderVAE, self).__init__()
            self.lstm = nn.LSTM(smiles_input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
            self.fc_mu = nn.Linear(latent_dim, latent_dim)  # 双向LSTM的输出需要乘以2
            self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        def forward(self, smiles):
            _, (h, _) = self.lstm(smiles)
            h = torch.cat((h[-2], h[-1]), dim=1)  # 拼接双向LSTM的最后一层输出

            residual = h
            mu = self.fc_mu(h)
            mu = mu + residual
            logvar = self.fc_logvar(h)
            logvar = logvar + residual
            return mu, logvar

    class DecoderVAE(nn.Module):
        def __init__(self,
                     latent_dim=512,
                     hidden_size=512,
                     output_size=3609):
            super(VAE_V3.DecoderVAE, self).__init__()
            self.fc1 = nn.Linear(latent_dim, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size)
            self.fc4 = nn.Linear(hidden_size, output_size)
            self.sigmoid = nn.Sigmoid()
            self.relu = nn.ReLU()

        def forward(self, z):
            # 第一层和第二层之间的残差连接
            residual = z
            h = self.relu(self.fc1(z))
            h = h + residual

            # 第二层和第三层之间的残差连接
            residual = h
            h = self.relu(self.fc2(h))
            h = h + residual

            # 第三层和第四层之间的残差连接
            residual = h
            h = self.relu(self.fc3(h))
            h = h + residual

            # 最后一层没有残差连接
            output = self.fc4(h)
            return self.sigmoid(output)

    def forward(self, smiles):
        mu, logvar = self.encoder(smiles)
        z = self.reparameterization(mu, logvar)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, logvar