import torch
import torch.nn as nn


class PhysicsInformedLoss(nn.Module):
    """联合偏微分方程约束的全局损失函数"""

    def __init__(self):
        super().__init__()
        self.lambda_phy = 0.1  # 物理约束调节系数
        self.a_max = 12.0  # 质心瞬时加速度极限 (12.0 m/s^2)
        self.mse = nn.MSELoss()

    def forward(self, pred, target, X_seq):
        # 1. 基础回归数据损失[cite: 1]
        loss_data = self.mse(pred, target)

        # 2. 运动学偏微分方程(PDE)损失计算[cite: 1]
        # 获取防守方(index 0)躯干质心(假设index 0为质心节点)轨迹[cite: 2]
        com = X_seq[:, :, 0, 0, :]  # shape: (B, T, 3)

        # 通过二阶导数计算局部加速度[cite: 2]
        v = com[:, 1:] - com[:, :-1]
        a = v[:, 1:] - v[:, :-1]
        a_norm = torch.norm(a, dim=-1)

        # 对超出 a_max 的极端物理发散施加惩罚[cite: 1, 2]
        penalty_acc = torch.relu(a_norm - self.a_max) ** 2

        return loss_data + self.lambda_phy * penalty_acc.mean()