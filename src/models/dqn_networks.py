import torch
import torch.nn as nn
import torch.nn.functional as F

class SailingQNetwork(nn.Module):
    """
    Neural Network for Deep Q-Learning in the sailing environment.
    Designed to process local boat state and wind conditions.
    """
    def __init__(self, state_size=6, action_size=9, hidden_size=128):
        """
        Initialize the network.
        
        Args:
            state_size (int): Dimension of input features (default 6: pos, vel, wind).
            action_size (int): Number of possible directions (default 9).
            hidden_size (int): Number of neurons in hidden layers.
        """
        super(SailingQNetwork, self).__init__()
        
        # Layer 1: Input to first hidden layer
        self.fc1 = nn.Linear(state_size, hidden_size)
        
        # Layer 2: Deeper representation of sailing physics
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Layer 3: Output layer (one Q-value per action)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        """
        Maps state to action values (Q-values).
        
        Args:
            state (torch.Tensor): The normalized local observation vector.
        """
        # ReLU activation allows the network to model non-linear sailing efficiency
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # No activation on the final layer; we want raw Q-value scores
        return self.fc3(x)