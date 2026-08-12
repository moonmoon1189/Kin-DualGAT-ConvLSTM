# data/transform.py
import torch


def inject_spatial_loss(X_seq, missing_ratio=0.20):
    """
    注入20%的空间坐标缺失掩码[cite: 1, 2]
    """
    mask = torch.rand_like(X_seq) > missing_ratio
    return X_seq * mask.float()


def inject_gaussian_noise(X_seq, snr_db=10):
    """
    注入特定信噪比(SNR=10dB)的高斯白噪声[cite: 1, 2]
    """
    signal_power = torch.mean(X_seq ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = torch.randn_like(X_seq) * torch.sqrt(noise_power)
    return X_seq + noise