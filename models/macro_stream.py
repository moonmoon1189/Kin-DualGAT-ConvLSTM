import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class MacroStream(nn.Module):
    """宏观流: 提取全局拓扑对抗特征[cite: 2]"""

    def __init__(self, in_channels=3, hidden_dim=128):
        super().__init__()
        self.sigma_d = 1.5
        self.gat1 = GATConv(in_channels, hidden_dim, heads=4, concat=False)  # 第一层[cite: 2]
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=8, concat=False)  # 第二层[cite: 2]

    def forward(self, x_def, x_off, edge_index):
        B, T, J, C = x_def.shape
        out = torch.zeros(B, T, J, 128, device=x_def.device)

        for b in range(B):
            for t in range(T):
                def_t = x_def[b, t]  # 防守者当前帧 (25, 3)
                off_t = x_off[b, t]  # 进攻者当前帧

                # 依据欧氏距离估算物理空间排斥势能[cite: 1, 2]
                d = def_t.unsqueeze(1) - off_t.unsqueeze(0)
                U = torch.exp(-(torch.norm(d, dim=-1) ** 2) / (2 * self.sigma_d ** 2))

                # 将势能转换为图的动态自适应边权重[cite: 2]
                edge_weight = torch.log(1 + U.mean(dim=1) + 1e-6)
                edge_attr = edge_weight[edge_index[0]].unsqueeze(-1)

                # 图特征聚合[cite: 2]
                h = torch.relu(self.gat1(def_t, edge_index, edge_attr=edge_attr))
                h = self.gat2(h, edge_index, edge_attr=edge_attr)
                out[b, t] = h

        return out  # 输出形状: (B, T, 25, 128)[cite: 2]