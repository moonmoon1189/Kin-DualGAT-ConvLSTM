import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2, lambda_phy=1.0):
        super(DynamicGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lambda_phy = lambda_phy

        # 对应方程8中的 W_v 和 W_a
        self.W_v = nn.Linear(in_features, out_features, bias=False)
        self.W_a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.W_v.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h_def, h_off, interaction_field):
        """
        h_def: 隐状态 (B, T, N, in_features) - 防守者
        h_off: 隐状态 (B, T, N, in_features) - 进攻者
        interaction_field: U_t^{ij} 动态交互场 (B, T, N, N)
        """
        Wh_def = self.W_v(h_def)  # (B, T, N, out_features)
        Wh_off = self.W_v(h_off)  # (B, T, N, out_features)

        B, T, N, E = Wh_def.size()

        # 特征拼接: W_v * X_def || W_v * X_off
        Wh_def_expanded = Wh_def.unsqueeze(3).expand(B, T, N, N, E)
        Wh_off_expanded = Wh_off.unsqueeze(2).expand(B, T, N, N, E)
        a_input = torch.cat([Wh_def_expanded, Wh_off_expanded], dim=-1)  # (B, T, N, N, 2E)

        # 基础注意力 Logit
        e = self.leakyrelu(torch.matmul(a_input, self.W_a).squeeze(-1))  # (B, T, N, N)

        # 方程8: 注入物理潜能项 \lambda * ln(1 + U_t^{ij} + \epsilon)
        epsilon = 1e-6
        physical_modulation = self.lambda_phy * torch.log(1.0 + interaction_field + epsilon)

        # 注意力权重计算 (Softmax)
        attention = F.softmax(e + physical_modulation, dim=-1)

        # 特征聚合
        h_prime = torch.matmul(attention, Wh_off)
        return F.elu(h_prime)