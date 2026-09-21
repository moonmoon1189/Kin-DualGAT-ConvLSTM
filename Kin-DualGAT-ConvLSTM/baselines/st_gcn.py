import torch
import torch.nn as nn


class STGCNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(6, 128, kernel_size=1)
        self.reg = nn.Linear(128, 1)

    def forward(self, x_def, x_off, **kwargs):
        x = torch.cat([x_def, x_off], dim=-1)  # B, T, J, 6
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x).mean(dim=(2, 3))
        return self.reg(x), None, None