# data/preprocess.py
import torch
import math


def global_spatial_anchor(X_raw, R, T):
    """
    全局空间锚定: 将相对相机坐标系映射至绝对的三维物理空间
    公式: X = R * X_raw + T
    """
    # X_raw shape: (T, 25, 3), R shape: (3, 3), T shape: (3,)
    X_anchored = torch.matmul(X_raw, R.T) + T
    return X_anchored


def temporal_gaussian_smooth(X_seq, tau=2, sigma=1.0):
    """
    时域高斯核平滑滤波: 消除传感器高频抖动噪声[cite: 1, 2]
    """
    T_seq, J, C = X_seq.shape
    X_smoothed = torch.zeros_like(X_seq)

    # 构建高斯核权重[cite: 2]
    weights = []
    for k in range(-tau, tau + 1):
        w = (1 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(-(k ** 2) / (2 * sigma ** 2))
        weights.append(w)
    weights = torch.tensor(weights, dtype=torch.float32, device=X_seq.device)
    weights = weights / weights.sum()  # 归一化[cite: 1]

    # 滑动窗口应用高斯滤波[cite: 1]
    for t in range(T_seq):
        t_start = max(0, t - tau)
        t_end = min(T_seq - 1, t + tau)

        # 截取有效窗口并获取对应权重
        valid_window = X_seq[t_start:t_end + 1]
        valid_weights = weights[t_start - (t - tau): t_end - (t - tau) + 1]
        valid_weights = valid_weights / valid_weights.sum()

        # 维度对齐并加权求和
        valid_weights = valid_weights.view(-1, 1, 1)
        X_smoothed[t] = (valid_window * valid_weights).sum(dim=0)

    return X_smoothed