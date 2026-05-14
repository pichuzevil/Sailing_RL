import torch
import torch.nn as nn
import torch.nn.functional as F

class SailingDuelingQNetwork(nn.Module):
    def __init__(self, state_size=10, action_size=9, hidden_size=256):
        super(SailingDuelingQNetwork, self).__init__()
        
        # Shared feature extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # State Value stream (V)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Action Advantage stream (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )

    def forward(self, state):
        features = self.feature_layer(state)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine using the stable dueling formula
        return value + (advantages - advantages.mean(dim=1, keepdim=True))