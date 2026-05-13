import os
import numpy as np
import torch
# Use local import for training; change to 'evaluator.base_agent' for Codabench
from agents.base_agent import BaseAgent 
from models.dqn_networks import SailingDuelingQNetwork

class DQNAgent(BaseAgent):
    def __init__(self, state_size=10, action_size=9, epsilon=0.1, weights_path=None):
        """
        Optimized DQN Agent for sailing navigation.
        STATE_SIZE should be 10 to include relative navigation features.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.epsilon = epsilon
        
        # 1. Initialize Networks with the CORRECT state size (10)
        self.policy_net = SailingDuelingQNetwork(state_size, action_size).to(self.device)
        self.target_net = SailingDuelingQNetwork(state_size, action_size).to(self.device)
        
        # 2. Load existing weights
        if weights_path is None:
            from utils.paths import get_dqn_save_path
            weights_path = get_dqn_save_path()

        if os.path.exists(weights_path):
            try:
                self.policy_net.load_state_dict(torch.load(weights_path, map_location=self.device))
                print(f"✅ DQNAgent: Resuming from {weights_path}")
            except Exception as e:
                print(f"⚠️ DQNAgent: Could not load weights ({e}). Starting fresh.")
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() 

    def preprocess_obs(self, observation, goal_pos=(110, 10)):
        """
        Transforms raw observation into a high-signal 10-feature vector.
        Features: [x, y, vx, vy, wx, wy, dx, dy, dist, angle_to_goal]
        """
        # Ensure we don't modify the original array
        raw_obs = observation[:6].copy()
        
        # 1. Normalization (Keeps gradients stable)
        raw_obs[0:2] /= 128.0  # x, y coords
        raw_obs[2:4] /= 8.0    # boat velocity
        raw_obs[4:6] /= 10.0   # wind velocity
        
        # 2. Relative Navigation (The "Compass")
        # How far is the goal?
        dx = (goal_pos[0] - observation[0]) / 128.0
        dy = (goal_pos[1] - observation[1]) / 128.0
        dist = np.sqrt(dx**2 + dy**2)
        
        # What is the angle to the goal? (Normalized between -1 and 1)
        angle_to_goal = np.arctan2(dy, dx) / np.pi
        
        # Return the 10-feature vector
        return np.concatenate([raw_obs, [dx, dy, dist, angle_to_goal]]).astype(np.float32)

    def act_from_vec(self, state_vec):
        """
        Used during TRAINING. Takes the 10-feature vector directly.
        """
        state = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
        
        # We use eval() here to ensure BatchNorm/Dropout (if any) don't update
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()

        if np.random.random() > self.epsilon:
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return np.random.choice(np.arange(self.action_size))

    def act(self, observation):
        """
        Used during EVALUATION/SUBMISSION. 
        Takes raw observation and handles its own preprocessing.
        """
        state_vec = self.preprocess_obs(observation)
        # Epsilon is 0 during evaluation
        state = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
        
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        
        return int(np.argmax(action_values.cpu().data.numpy()))

    def reset(self):
        pass

    def seed(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)