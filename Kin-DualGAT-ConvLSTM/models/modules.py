import torch
import torch.nn as nn
import torch.nn.functional as F


class InteractionField(nn.Module):
    def __init__(self, eps=1e-6, r_cyl=1.0, sigma_d=1.5, gamma=5.0):
        super().__init__()
        self.eps = eps
        self.r_cyl = r_cyl
        self.sigma_d = sigma_d
        self.gamma = gamma

    def forward(self, d_t, v_t):
        # Equation 7 implementation
        d_norm = torch.norm(d_t, dim=-1) + self.eps
        dot_product = torch.sum(v_t * d_t, dim=-1)

        term1 = torch.exp(-(d_norm ** 2) / (2 * self.sigma_d ** 2))
        term2 = F.relu(-(dot_product) / d_norm + self.eps)
        term3 = 1.0 / (1.0 + torch.exp(self.gamma * (d_norm - self.r_cyl)))

        U_t = term1 * term2 * term3
        return U_t.unsqueeze(-1)


class PosturalAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, cov_matrix):
        # cov_matrix shape approx: [B, T, J, 3, 3] -> simulate eigenvalues for attention
        eigen_approx = torch.diagonal(cov_matrix, dim1=-2, dim2=-1)  # [B, T, J, 3]
        l1, l2, l3 = eigen_approx[..., 0], eigen_approx[..., 1], eigen_approx[..., 2]

        # Equation 12: beta in [0, 1]
        beta = (l1 - l2) / (l1 + l2 + l3 + self.eps)
        return torch.clamp(beta, 0.0, 1.0).unsqueeze(-1)


class GraphToGrid(nn.Module):
    def __init__(self, in_channels, out_channels, grid_size=5):
        super().__init__()
        self.grid_size = grid_size
        self.num_joints = 25
        self.proj = nn.Linear(in_channels * self.num_joints, out_channels * grid_size * grid_size)

    def forward(self, x):
        # x: [B, T, J, C] -> [B, T, C, H, W]
        B, T, J, C = x.shape
        x = x.view(B, T, -1)
        x = self.proj(x)
        x = x.view(B, T, -1, self.grid_size, self.grid_size)
        return x


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        # Strictly uses 2D convolution for spatiotemporal states
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)

        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next


class Decoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_com = nn.Linear(hidden_dim, 3)
        self.fc_angles = nn.Linear(hidden_dim, 3)

    def forward(self, h):
        B, T, C, H, W = h.shape
        h_pool = self.pool(h.view(B * T, C, H, W)).view(B, T, C)
        com_traj = self.fc_com(h_pool)
        angles = self.fc_angles(h_pool)
        return com_traj, angles