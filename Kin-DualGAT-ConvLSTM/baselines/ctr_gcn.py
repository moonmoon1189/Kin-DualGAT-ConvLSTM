import torch
import torch.nn as nn


class CTRGCNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(6, 128, kernel_size=3, padding=1)
        self.reg = nn.Linear(128, 1)

    def forward(self, x_def, x_off, **kwargs):
        x = torch.cat([x_def, x_off], dim=-1).permute(0, 3, 1, 2)
        x = torch.relu(self.conv(x)).mean(dim=(2, 3))
        return self.reg(x), None, None