# eval.py
import torch
from torch.utils.data import DataLoader
from data.dataset import RealSkeletonDataset
from utils.skeleton_utils import get_edge_index
from utils.metrics import calculate_all_metrics
from models.macro_stream import MacroStream
from models.micro_stream import MicroStream
from models.phy_dualgat_convlstm import PhyDualGATConvLSTM

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"评估执行平台: {device}")

    # 1. 初始化模型并推入设备
    macro = MacroStream().to(device)
    micro = MicroStream().to(device)
    model = PhyDualGATConvLSTM(macro, micro).to(device)

    # 加载训练好的最佳模型权重
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    # 2. 加载测试集数据（使用拆分后的 test.xlsx）
    test_dataset = RealSkeletonDataset("data/test.xlsx", seq_len=156)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    edge_index = get_edge_index().to(device)

    all_preds = []
    all_targets = []

    print("开始在独立测试集上执行正向推演...")

    # 3. 关闭梯度计算，执行推理
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            pred = model(batch_X, edge_index)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    # 4. 计算并格式化输出所有评估指标
    results = calculate_all_metrics(all_targets, all_preds)
    print("\n" + "=" * 40)
    print("模型在独立测试集上的综合回归性能测试结果:")
    for k, v in results.items():
        print(f"{k:>6}: {v:.4f}")
    print("=" * 40)

if __name__ == "__main__":
    evaluate()