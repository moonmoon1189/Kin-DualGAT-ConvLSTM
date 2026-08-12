# visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.dataset import RealSkeletonDataset
from utils.skeleton_utils import get_edge_index
from models.macro_stream import MacroStream
from models.micro_stream import MicroStream
from models.phy_dualgat_convlstm import PhyDualGATConvLSTM

# ------------------- 1. 加载模型和数据 -------------------
def load_model_and_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    macro = MacroStream().to(device)
    micro = MicroStream().to(device)
    model = PhyDualGATConvLSTM(macro, micro).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    # 加载测试集
    test_dataset = RealSkeletonDataset("data/test.xlsx", seq_len=156)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    edge_index = get_edge_index().to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            pred = model(batch_X, edge_index)
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(batch_y.cpu().numpy().flatten())

    return np.array(all_targets), np.array(all_preds)

# ------------------- 2. 绘图函数 -------------------
def plot_residual_distribution(y_true, y_pred, save_path="residual_distribution.png"):
    """图4(b)：残差分布 + 长尾截断效果"""
    residuals = y_pred - y_true
    plt.figure(figsize=(10, 6))
    sns.kdeplot(residuals, fill=True, color="#4169E1", linewidth=2.5, alpha=0.6,
                label="Phy-DualGAT-ConvLSTM (Full)")

    # 这里因为没有 baseline，我们只画一条曲线。如果想画对比，需另外准备 baseline 预测
    # 可以模拟一个无物理约束的残差分布（仅作示意），或留空。
    # 为简单，我们添加物理边界线
    plt.axvline(x=0.1, color='red', linestyle='-.', linewidth=2, alpha=0.7)
    plt.axvline(x=-0.1, color='red', linestyle='-.', linewidth=2, alpha=0.7)
    plt.axvspan(0.1, 0.35, ymin=0, ymax=0.15, facecolor='red', alpha=0.1, hatch='//')
    plt.axvspan(-0.35, -0.1, ymin=0, ymax=0.15, facecolor='red', alpha=0.1, hatch='//')

    plt.title("Prediction Residual Distribution (Error Convergence)", fontsize=14)
    plt.xlabel("Prediction Residual: $Y_{pred} - Y_{true}$", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.xlim(-0.35, 0.35)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"残差分布图已保存至 {save_path}")

def plot_bland_altman(y_true, y_pred, save_path="bland_altman.png"):
    """图4(a)：Bland-Altman 一致性限度图"""
    mean = (y_true + y_pred) / 2
    diff = y_pred - y_true
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    plt.figure(figsize=(10, 6))
    plt.scatter(mean, diff, alpha=0.5, s=10, color='blue')
    plt.axhline(mean_diff, color='black', linestyle='-', linewidth=2, label=f'Mean diff = {mean_diff:.3f}')
    plt.axhline(loa_upper, color='gray', linestyle='--', linewidth=2, label=f'LoA +1.96SD = {loa_upper:.3f}')
    plt.axhline(loa_lower, color='gray', linestyle='--', linewidth=2, label=f'LoA -1.96SD = {loa_lower:.3f}')

    plt.xlabel("Average of True and Predicted", fontsize=12)
    plt.ylabel("Difference (Predicted - True)", fontsize=12)
    plt.title("Bland-Altman Plot for Consistency Analysis", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Bland-Altman 图已保存至 {save_path}")

def plot_tyler_diagram(y_true, y_pred, save_path="tyler_diagram.png"):
    """
    泰勒图：展示标准差、相关系数和中心化均方根误差。
    这里简化：只绘制当前模型的一个点，若要对比多个模型，可扩展。
    """
    # 计算指标
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    # 归一化标准差（相对于真实值）
    norm_std = std_pred / std_true

    # 绘制极坐标泰勒图（简化）
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    # 角度对应相关系数（arccos）
    theta = np.arccos(corr)
    r = norm_std
    ax.scatter(theta, r, s=100, c='red', marker='o', label='Phy-DualGAT-ConvLSTM')

    # 添加参考弧线
    r_ticks = np.arange(0.5, 1.6, 0.25)
    ax.set_rlim(0, 1.6)
    ax.set_rticks(r_ticks)
    ax.set_rlabel_position(135)

    # 添加相关系数等值线（简化）
    ax.set_title("Taylor Diagram", fontsize=14)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"泰勒图已保存至 {save_path}")

# ------------------- 3. 主函数 -------------------
def main():
    print("正在加载模型并对测试集进行预测...")
    y_true, y_pred = load_model_and_predict()

    print(f"共获取 {len(y_true)} 个样本")
    print(f"预测均值: {np.mean(y_pred):.4f}, 标准差: {np.std(y_pred):.4f}")
    print(f"真实均值: {np.mean(y_true):.4f}, 标准差: {np.std(y_true):.4f}")

    # 生成所有图表
    plot_residual_distribution(y_true, y_pred, "residual_distribution.png")
    plot_bland_altman(y_true, y_pred, "bland_altman.png")
    plot_tyler_diagram(y_true, y_pred, "tyler_diagram.png")

    print("所有可视化图表生成完毕！")

if __name__ == "__main__":
    main()