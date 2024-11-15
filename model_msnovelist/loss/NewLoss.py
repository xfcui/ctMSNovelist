import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class NewLoss(nn.Module):
    def __init__(self):
        super(NewLoss, self).__init__()

    def forward(self, preds, labels):
        eps = 1e-8  # 防止数值超出定义域
        loss_y1_part1 = (-torch.log(preds + eps) + 1.5 + torch.log(torch.tensor(0.5))) * labels
        loss_y1_part2 = 2 * (1 - preds ** 2) * labels

        loss_y0_part1 = (-torch.log(1 - preds + eps) + 1.5 + torch.log(torch.tensor(0.5))) * (1-labels)
        loss_y0_part2 = (4 * preds - 2 * preds ** 2) * (1-labels)

        loss_y1 = torch.where(preds < 0.5, loss_y1_part1, loss_y1_part2)
        loss_y0 = torch.where((1-preds) < 0.5, loss_y0_part1, loss_y0_part2)
        loss = loss_y1 + loss_y0
        return torch.mean(loss)


if __name__ == "__main__":
    for i in range(11):
        a = i/10
        b = 1-a
        preds = torch.tensor([1, b], dtype=torch.float32)
        labels = torch.tensor([1, 0], dtype=torch.float32)
        criterion = NewLoss()
        loss = criterion(preds, labels)
        print(f"Focal Loss: {loss.item()}")
    preds = torch.tensor([0.5, 0.5], dtype=torch.float32)
    # preds = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    # labels = torch.tensor([0.9, 0.8, 0.1, 0.2], dtype=torch.float32)

