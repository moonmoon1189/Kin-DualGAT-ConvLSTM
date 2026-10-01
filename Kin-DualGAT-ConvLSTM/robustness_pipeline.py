import numpy as np
from scipy.interpolate import CubicSpline


def apply_robustness_perturbation(skeleton_seq, missing_rate=0.2, snr_db=10):
    """
    skeleton_seq: (T, N, 3) 原始连续骨骼序列
    缺失坐标并进行三次样条插值，同时加入等效于特定 SNR 的高斯白噪声。
    """
    T, N, D = skeleton_seq.shape
    corrupted_seq = skeleton_seq.copy()

    # --- 1. 高斯白噪声注入 (SNR 基于去中心化轨迹计算) ---
    if snr_db is not None:
        # 去中心化计算真实信号功率，避免原点依赖
        mean_centered_seq = corrupted_seq - np.mean(corrupted_seq, axis=0, keepdims=True)
        signal_power = np.mean(mean_centered_seq ** 2)

        # 根据 SNR 反推噪声功率
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power)

        # 生成并加入高斯白噪声
        noise = np.random.normal(0, noise_std, size=(T, N, D))
        corrupted_seq += noise

    # --- 2. 随机坐标缺失与三次样条插值 ---
    if missing_rate > 0.0:
        for n in range(N):
            for d in range(D):
                # 生成缺失掩码
                mask = np.random.rand(T) > missing_rate

                # 如果全部缺失或保留帧过少，跳过插值以防崩溃
                valid_indices = np.where(mask)[0]
                if len(valid_indices) < 4:
                    continue

                    # 提取保留点并利用三次样条插值填补缺失帧
                valid_values = corrupted_seq[valid_indices, n, d]
                cs = CubicSpline(valid_indices, valid_values)

                all_indices = np.arange(T)
                corrupted_seq[:, n, d] = cs(all_indices)

    return corrupted_seq

# 示例: corrupted_skeleton = apply_robustness_perturbation(original_skeleton, missing_rate=0.2, snr_db=10)