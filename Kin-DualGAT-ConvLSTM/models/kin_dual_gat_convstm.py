import torch
import torch.nn as nn
from models.modules import InteractionField, PosturalAttention, GraphToGrid, ConvLSTMCell, Decoder


class KinDualGATConvLSTM(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feat_extractor = nn.Linear(in_channels, hidden_dim)

        self.interaction_field = InteractionField()
        self.postural_attn = PosturalAttention()

        self.graph_to_grid = GraphToGrid(hidden_dim * 2, hidden_dim)
        self.convlstm = ConvLSTMCell(hidden_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim)

        self.regression = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 5, 5)),
            nn.Flatten(),
            # 【修复点】：展平后的维度必须是 hidden_dim * 5 * 5 = hidden_dim * 25
            nn.Linear(hidden_dim * 25, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def extract_macro_features(self, f_def, f_off, x_def, x_off):
        v_def = torch.zeros_like(x_def)
        v_def[:, 1:] = x_def[:, 1:] - x_def[:, :-1]
        v_off = torch.zeros_like(x_off)
        v_off[:, 1:] = x_off[:, 1:] - x_off[:, :-1]

        d_t = x_def - x_off
        v_t = v_def - v_off
        U_t = self.interaction_field(d_t, v_t)

        return f_def * U_t + f_off

    def extract_micro_features(self, f_def, x_def):
        B, T, J, C = x_def.shape
        x_mean = x_def.mean(dim=1, keepdim=True)
        x_centered = x_def - x_mean
        cov_matrix = torch.matmul(x_centered.unsqueeze(-1), x_centered.unsqueeze(-2))

        beta = self.postural_attn(cov_matrix)
        return f_def * beta

    def fuse_features(self, f_macro, f_micro):
        return torch.cat([f_macro, f_micro], dim=-1)

    def temporal_evolution(self, grid_seq):
        B, T, C, H, W = grid_seq.shape
        device = grid_seq.device
        h = torch.zeros(B, self.hidden_dim, H, W).to(device)
        c = torch.zeros(B, self.hidden_dim, H, W).to(device)

        outputs = []
        for t in range(T):
            h, c = self.convlstm(grid_seq[:, t], h, c)
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)

    def forward(self, x_def, x_off, A_adj=None):
        f_def = self.feat_extractor(x_def)
        f_off = self.feat_extractor(x_off)

        f_macro = self.extract_macro_features(f_def, f_off, x_def, x_off)
        f_micro = self.extract_micro_features(f_def, x_def)

        f_fused = self.fuse_features(f_macro, f_micro)
        grid_seq = self.graph_to_grid(f_fused)

        h_seq = self.temporal_evolution(grid_seq)

        com_traj, angles = self.decoder(h_seq)
        # 将 [B, T, C, H, W] 转为 [B, C, T, H, W] 给 3D 池化
        score = self.regression(h_seq.permute(0, 2, 1, 3, 4))

        return score, com_traj, angles