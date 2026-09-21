import torch
import torch.nn as nn

class UnregisteredRawBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(25 * 3 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def concatenate(self, x_def, x_off):
        # Skip spatial registration, directly concat
        B, T, J, C = x_def.shape
        x = torch.cat([x_def, x_off], dim=-1).view(B, T, -1)
        return x

    def forward(self, x_def, x_off, **kwargs):
        x = self.concatenate(x_def, x_off)
        x_mean = x.mean(dim=1) # Average over time
        return self.mlp(x_mean), None, None