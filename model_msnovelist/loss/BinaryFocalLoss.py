import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        eps = 1e-7  # 防止数值超出定义域
        # 计算 focal loss
        loss_y1 = -1 * self.alpha * torch.pow((1 - preds), self.gamma) * torch.log(preds + eps) * labels
        loss_y0 = -1 * (1 - self.alpha) * torch.pow(preds, self.gamma) * torch.log(1 - preds + eps) * (1 - labels)
        loss = loss_y0 + loss_y1
        return torch.mean(loss)


if __name__ == "__main__":
    # 假设我们有一个简单的二分类问题
    # preds = torch.tensor([0.9, 0.8, 0.1, 0.4], dtype=torch.float32)
    preds = torch.tensor([[1, 1, 0, 0],
                          [1, 1, 0, 0]], dtype=torch.float32)
    # labels = torch.tensor([0.9, 0.8, 0.1, 0.4], dtype=torch.float32)
    labels = torch.tensor([[1, 1, 0, 0],
                           [1, 1, 0, 0]], dtype=torch.float32)

    # 创建 Binary Focal Loss 实例
    criterion = BinaryFocalLoss(alpha=0.25, gamma=2)
    # 计算损失
    loss = criterion(preds, labels)
    print(f"Focal Loss: {loss.item()}")
    criterion2 = nn.BCELoss()
    print(criterion2(preds, labels))
