import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def participant_clustered_bootstrap(y_true, y_pred, participant_ids, n_iterations=5000):
    """
    基于参与者级别的 Bootstrap 置信区间计算。
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    participant_ids = np.array(participant_ids)

    unique_participants = np.unique(participant_ids)
    n_participants = len(unique_participants)

    metrics = {'R2': [], 'RMSE': [], 'MAE': [], 'PLCC': [], 'SRCC': []}

    for _ in range(n_iterations):
        # 有放回地重采样参与者 ID
        resampled_pids = np.random.choice(unique_participants, size=n_participants, replace=True)

        # 提取重采样后的序列索引
        resampled_indices = []
        for pid in resampled_pids:
            resampled_indices.extend(np.where(participant_ids == pid)[0])

        y_t_boot = y_true[resampled_indices]
        y_p_boot = y_pred[resampled_indices]

        # 计算指标
        metrics['R2'].append(r2_score(y_t_boot, y_p_boot))
        metrics['RMSE'].append(np.sqrt(mean_squared_error(y_t_boot, y_p_boot)))
        metrics['MAE'].append(mean_absolute_error(y_t_boot, y_p_boot))
        metrics['PLCC'].append(pearsonr(y_t_boot, y_p_boot)[0])
        metrics['SRCC'].append(spearmanr(y_t_boot, y_p_boot)[0])

    results = {}
    for k, v in metrics.items():
        v = np.array(v)
        results[k] = {
            'mean': np.mean(v),
            'ci_lower': np.percentile(v, 2.5),
            'ci_upper': np.percentile(v, 97.5)
        }

    return results

# 示例调用
# res = participant_clustered_bootstrap(labels, preds, p_ids)
# print(f"R2: {res['R2']['mean']:.3f} [{res['R2']['ci_lower']:.3f}, {res['R2']['ci_upper']:.3f}]")