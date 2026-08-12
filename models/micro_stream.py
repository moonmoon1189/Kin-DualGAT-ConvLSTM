import torch
import torch.nn as nn


class MicroStream(nn.Module):
    """微观流: 剥离并量化微观生物力学属性[cite: 2]"""

    def __init__(self, hidden_dim=128, window=5):
        super().__init__()
        self.W = window  # 局部时间观测窗[cite: 2]
        self.conv1x1 = nn.Conv2d(3, hidden_dim, kernel_size=1)  # 特征降维映射[cite: 2]

    def forward(self, x_def):
        B, T, J, C = x_def.shape
        out = torch.zeros(B, T, J, 128, device=x_def.device)

        for t in range(T):
            # 获取局部时间窗数据[cite: 2]
            t_start = max(0, t - self.W // 2)
            t_end = min(T, t + self.W // 2 + 1)
            window = x_def[:, t_start:t_end, :, :]

            # 计算局部协方差矩阵[cite: 2]
            mu = window.mean(dim=1, keepdim=True)
            diff = window - mu
            cov = torch.einsum('bwjc,bwjd->bjcd', diff, diff) / self.W

            # 奇异值(SVD)分解以提取运动主成分特征根[cite: 1, 2]
            cov = cov + torch.eye(3, device=cov.device) * 1e-6  # 增加数值垫防止计算崩溃
            U, S, Vh = torch.linalg.svd(cov)

            # 求解空间各向异性的局部姿态注意力权重[cite: 1, 2]
            lam1, lam2, lam3 = S[..., 0], S[..., 1], S[..., 2]
            beta = (lam1 - lam2) / (lam1 + lam2 + lam3 + 1e-6)
            beta = beta.unsqueeze(-1).unsqueeze(-1)

            # 对核心发力部位进行注意力加权[cite: 2]
            weighted = x_def[:, t:t + 1, :, :].permute(0, 3, 2, 1) * beta.permute(0, 3, 1, 2)
            out[:, t] = self.conv1x1(weighted).squeeze(-1).permute(0, 2, 1)

        return out  # 输出形状: (B, T, 25, 128)[cite: 2]