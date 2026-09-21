import torch
import torch.nn as nn

class TraditionalICPBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x_def, x_off, **kwargs):
        # Uses purely centroid Euclidean distances, no interaction fields
        c_def = x_def.mean(dim=2)
        c_off = x_off.mean(dim=2)
        dist = torch.norm(c_def - c_off, dim=-1).mean(dim=1, keepdim=True)
        return self.mlp(dist), None, None