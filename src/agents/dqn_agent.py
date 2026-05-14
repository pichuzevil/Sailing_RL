import os
import numpy as np
import torch
try:
    from agents.base_agent import BaseAgent 
except ImportError:
    from evaluator.base_agent import BaseAgent

from models.dqn_networks import SailingDuelingQNetwork

class DQNAgent(BaseAgent):
    def __init__(self, state_size=11, action_size=9, epsilon=0.1, weights_path=None):
        """
        Improved DQN Agent with 11-feature state vector.
        Feature 11: Static Island Proximity (Radar).
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.epsilon = epsilon
        
        # State size is now 11 (6 base + 4 navigation + 1 island radar)
        self.policy_net = SailingDuelingQNetwork(state_size, action_size).to(self.device)
        self.target_net = SailingDuelingQNetwork(state_size, action_size).to(self.device)
        
        if weights_path and os.path.exists(weights_path):
            try:
                self.policy_net.load_state_dict(torch.load(weights_path, map_location=self.device))
                print(f"✅ DQNAgent: Loaded weights from {weights_path}")
            except Exception as e:
                print(f"⚠️ DQNAgent: Could not load weights ({e}). Starting fresh.")
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() 

    def preprocess_obs(self, observation, goal_pos=(64, 127)):
        """
        Transforms raw observation into a high-signal 11-feature vector.
        Feature 11 is now a Geometric Radar that respects island corners.
        """
        # 1. Base 6 features (Normalization)
        obs = observation[:6].copy()
        obs[0:2] /= 128.0  # x, y
        obs[2:4] /= 8.0    # vx, vy
        obs[4:6] /= 10.0   # wx, wy
        
        # 2. Relative Navigation (Goal)
        dx_g = (goal_pos[0] - observation[0]) / 128.0
        dy_g = (goal_pos[1] - observation[1]) / 128.0
        dist_g = np.sqrt(dx_g**2 + dy_g**2)
        angle_g = np.arctan2(dy_g, dx_g) / np.pi 
        
        # 3. GEOMETRIC ISLAND RADAR (Feature 11)
        x, y = observation[0], observation[1]
        
        # Exact distance to the Rectangle body: X[38, 90], Y[43, 85]
        dx_rect = max(38 - x, 0, x - 90)
        dy_rect = max(43 - y, 0, y - 85)
        dist_rect = np.sqrt(dx_rect**2 + dy_rect**2)
        
        # Exact distance to the Triangle tip: (64, 17)
        # We approximate the southern reach to ensure the point is avoided
        dx_tri = max(38 - x, 0, x - 90)
        dy_tri = max(17 - y, 0, y - 43)
        dist_tri = np.sqrt(dx_tri**2 + dy_tri**2)
        
        # Final raw distance to the nearest part of the house
        raw_dist = min(dist_rect, dist_tri)
        
        # Normalize Radar signal: 
        # 0.0 = Right on the edge/corner, 1.0 = Safe (>20 pixels away)
        proximity_radar = np.clip(raw_dist / 20.0, 0, 1)
        
        return np.concatenate([
            obs, 
            [dx_g, dy_g, dist_g, angle_g], 
            [proximity_radar]
        ]).astype(np.float32)

    def act_from_vec(self, state_vec):
        """Used during TRAINING."""
        state = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        
        if np.random.random() > self.epsilon:
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return np.random.choice(np.arange(self.action_size))

    def act(self, observation):
        """Used during EVALUATION/SUBMISSION."""
        state_vec = self.preprocess_obs(observation)
        state = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
        
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        
        return int(np.argmax(action_values.cpu().data.numpy()))