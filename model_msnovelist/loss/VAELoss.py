import torch
from torch import nn


class VAELoss(nn.Module):
    def __init__(self, recon_loss_type='bce'):
        super(VAELoss, self).__init__()
        self.recon_loss_type = recon_loss_type.lower()

        if self.recon_loss_type == 'bce':
            self.reconstruction_loss = nn.BCELoss(reduction='sum')
        elif self.recon_loss_type == 'mse':
            self.reconstruction_loss = nn.MSELoss(reduction='sum')
        else:
            raise ValueError("Unsupported reconstruction loss type. Use 'bce' or 'mse'.")

    def forward(self, recon_x, x, mu, logvar):
        if self.recon_loss_type == 'bce':
            recon_loss = self.reconstruction_loss(recon_x, x)
        elif self.recon_loss_type == 'mse':
            recon_loss = self.reconstruction_loss(recon_x, x)

        # KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # print(f"recon_loss: {recon_loss:.2f}, kl_loss: {kl_loss:.2f}")
        return recon_loss + kl_loss