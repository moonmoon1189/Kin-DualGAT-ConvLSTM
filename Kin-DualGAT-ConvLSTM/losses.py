import torch
import torch.nn as nn


def kinematic_constraint_loss(com_traj, angles, mask, a_max=12.0):
    # com_traj: [B, T, 3] -> v_t: [B, T-1, 3] -> a_t: [B, T-2, 3]
    v_t = com_traj[:, 1:] - com_traj[:, :-1]
    a_t = v_t[:, 1:] - v_t[:, :-1]

    a_norm = torch.norm(a_t, dim=-1)
    a_penalty = torch.relu(a_norm - a_max) ** 2

    # 论文 Table 2 阈值，转为弧度制
    theta_max = torch.tensor([125.0, 145.0, 50.0], device=angles.device) * (3.14159 / 180.0)
    # 统一丢弃前两帧以对齐时间步长度 T-2
    angle_penalty = torch.relu(torch.abs(angles[:, 2:]) - theta_max) ** 2

    # 【修复点】确保 mask_a 的形状为 [B, T-2]
    mask_a = mask[:, 2:].float()

    loss_a = (a_penalty * mask_a).sum() / (mask_a.sum() + 1e-6)
    loss_angle = (angle_penalty.sum(-1) * mask_a).sum() / (mask_a.sum() + 1e-6)

    return loss_a + loss_angle


def total_loss(pred_y, true_y, com_traj, angles, length, lambda_phy=0.1):
    B, T = com_traj.shape[:2]
    device = pred_y.device

    mask = torch.arange(T, device=device).expand(B, T) < length.unsqueeze(1)

    mse = nn.MSELoss()(pred_y.view(-1), true_y.view(-1))
    phy = kinematic_constraint_loss(com_traj, angles, mask)

    return mse + lambda_phy * phy