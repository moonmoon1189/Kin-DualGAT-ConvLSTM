# utils/metrics.py
import numpy as np
from scipy.stats import pearsonr, spearmanr


def calculate_all_metrics(y_true, y_pred):
    """
    计算论文表5中的各项核心回归指标
    输入参数为真实标签序列与模型预测序列
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # 1. 均方根误差 (RMSE) 与 平均绝对误差 (MAE)[cite: 2]
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    # 2. 判定系数 (R^2)[cite: 2]
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    # 3. 皮尔逊 (PLCC) 与 斯皮尔曼 (SRCC) 相关系数[cite: 2]
    plcc, _ = pearsonr(y_true, y_pred)
    srcc, _ = spearmanr(y_true, y_pred)

    return {
        "R2": float(r2),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "PLCC": float(plcc),
        "SRCC": float(srcc)
    }