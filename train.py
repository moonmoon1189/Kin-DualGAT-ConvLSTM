# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import RealSkeletonDataset
from utils.skeleton_utils import get_edge_index
from models.macro_stream import MacroStream
from models.micro_stream import MicroStream
from models.phy_dualgat_convlstm import PhyDualGATConvLSTM
from losses.physics_loss import PhysicsInformedLoss
import math

def main():
    # 1. 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练平台: {device}")

    # 2. 加载训练集和验证集（已按 subject 划分）
    train_dataset = RealSkeletonDataset("data/train.xlsx", seq_len=156)
    val_dataset = RealSkeletonDataset("data/val.xlsx", seq_len=156)

    # 根据显存大小调整 batch_size，建议 16 或 32，数据量小可以设为 16
    batch_size = 16   # 若显存不足，可减为 8 或 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")

    # 3. 图结构
    edge_index = get_edge_index().to(device)

    # 4. 模型
    macro = MacroStream().to(device)
    micro = MicroStream().to(device)
    model = PhyDualGATConvLSTM(macro, micro).to(device)

    # 5. 优化器与损失函数（物理损失权重 lambda_phy=0.1 已在 PhysicsInformedLoss 中定义）
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = PhysicsInformedLoss().to(device)  # 内部 lambda_phy = 0.1

    # 6. 学习率调度：预热 + 余弦退火
    epochs = 150  # 最大轮次，早停会提前终止
    warmup_epochs = 10
    base_lr = 1e-3
    min_lr = 1e-6

    def get_lr(epoch):
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

    # 7. 早停参数
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20

    print("开始训练...")
    for epoch in range(epochs):
        # 更新学习率
        lr = get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ---------- 训练 ----------
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_X, edge_index)
            loss = criterion(pred, batch_y, batch_X)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        # ---------- 验证 ----------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                pred = model(batch_X, edge_index)
                loss = criterion(pred, batch_y, batch_X)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1:3d}/{epochs}] | LR: {lr:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ---------- 保存最佳模型 & 早停 ----------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
            print(f"  -> 验证损失降至 {best_val_loss:.4f}，已保存最佳模型")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发：验证损失连续 {patience} 轮未改善，训练终止。")
                break

    print(f"\n训练完成！最佳验证损失: {best_val_loss:.4f}")
    print("最佳模型权重已保存至 'best_model.pth'")

if __name__ == "__main__":
    main()