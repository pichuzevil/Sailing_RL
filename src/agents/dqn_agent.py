import os
import numpy as np
import torch
# Use local import for training; change to 'evaluator.base_agent' for Codabench
from agents.base_agent import BaseAgent 
from models.dqn_networks import SailingQNetwork

class DQNAgent(BaseAgent):
    def __init__(self, state_size=6, action_size=9, epsilon=0.1):
        """
        DQN Agent for sailing navigation.
        
        Args:
            state_size (int): Local observation size (pos, vel, wind).
            action_size (int): Number of directions (9).
            epsilon (float): Initial exploration rate.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.epsilon = epsilon
        
        # 1. Initialize Networks
        self.policy_net = SailingQNetwork(state_size, action_size).to(self.device)
        self.target_net = SailingQNetwork(state_size, action_size).to(self.device)
        
        # 2. Load existing weights if they exist (to resume training)
        weights_path = os.path.join(os.path.dirname(__file__), 'dqn_weights.pth')
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            self.policy_net.load_state_dict(state_dict)
            print(f"DQNAgent: Resuming from weights at {weights_path}")
        else:
            print("DQNAgent: Starting training from scratch.")

        # 3. Synchronize networks
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target net is only for inference labels
        

    def act(self, observation):
        """Standard epsilon-greedy action selection."""
        # Normalize local features
        local_obs = observation[:6].copy()
        local_obs[0:2] /= 128.0 
        
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
            
        state = torch.from_numpy(local_obs).float().unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()
        
        return int(np.argmax(action_values.cpu().data.numpy()))

    def reset(self):
        """Reset internal state if needed (not required for standard DQN)."""
        pass

    def seed(self, seed=None):
        """Set seeds for reproducibility."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)