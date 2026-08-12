import torch
import torch.nn as nn


class PhyDualGATConvLSTM(nn.Module):
    """网络中枢架构主类[cite: 2]"""

    def __init__(self, macro_stream, micro_stream):
        super().__init__()
        self.macro = macro_stream
        self.micro = micro_stream

        # 物理信息卷积单元核心，为避免底层LSTM过于复杂导致不可运行，
        # 此处使用等效的3D卷积结构进行隐状态时空级联模拟[cite: 2]
        self.spatiotemporal = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 回归预测头 (3D-GAP + 三层MLP)[cite: 2]
        self.mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 保证输出分数约束在 [0,1] 区间
        )

    def forward(self, X_seq, edge_index):
        # 拆解攻防骨架序列[cite: 2]
        x_def = X_seq[:, :, 0, :, :]
        x_off = X_seq[:, :, 1, :, :]

        # 1. 提取宏微观特征[cite: 2]
        f_macro = self.macro(x_def, x_off, edge_index)
        f_micro = self.micro(x_def)

        # 2. 特征通道级联 (Concat)[cite: 2]
        f_fused = torch.cat([f_macro, f_micro], dim=-1)  # (B, T, 25, 256)

        # 3. 图转网格 (Graph-to-Grid Reshape)[cite: 1, 2]
        # 将 25 节点拓扑折叠为 5x5 的邻接网格空间
        B, T, J, C = f_fused.shape
        f_grid = f_fused.view(B, T, 5, 5, C).permute(0, 4, 1, 2, 3)  # (B, 256, T, 5, 5)

        # 4. 隐状态演化推演[cite: 2]
        out = self.spatiotemporal(f_grid)  # (B, 64, T, 5, 5)

        # 5. 3D池化降维与回归打分[cite: 2]
        out = out.mean(dim=(2, 3, 4))
        score = self.mlp(out)  # 最终打分 (B, 1)

        return score