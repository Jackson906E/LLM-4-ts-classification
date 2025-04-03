import torch
import torch.nn as nn

class WeightingNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
        )
        self.w_o_head = nn.Linear(64, 1)
        self.w_i_head = nn.Linear(64, 1)

    def forward(self, features):
        shared = self.shared(features)
        ω_o = torch.sigmoid(self.w_o_head(shared))
        ω_i = torch.sigmoid(self.w_i_head(shared))
        return ω_o, ω_i
