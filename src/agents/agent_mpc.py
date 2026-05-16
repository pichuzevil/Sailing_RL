import numpy as np
from typing import Dict, Any, Optional
try:
    from agents.base_agent import BaseAgent 
except ImportError:
    from evaluator.base_agent import BaseAgent

class MPCAgent(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        # 1. OPTUNA CONFIGURATION
        self.cfg = config or {
            'horizon': 15,        # Steps to look ahead
            'num_samples': 40,    # Number of random trajectories per act()
            'vmg_weight': 1.2,    # Importance of Y-axis progress
            'safety_buffer': 2.5, # Padding around islands
            'dist_weight': 0.5    # Importance of total distance to goal
        }
        
        self.max_speed = 8.0
        self.inertia_factor = 0.3
        self.mean_rotation = 0.5
        self.goal_pos = np.array([64, 127])
        self.world_map = None

        self.actions = [np.array([0, 1]), np.array([1, 1]), np.array([1, 0]), np.array([1, -1]),
                        np.array([0, -1]), np.array([-1, -1]), np.array([-1, 0]), np.array([-1, 1]),
                        np.array([0, 0])]
        self.action_dirs = [a / np.linalg.norm(a) if np.any(a) else a for a in self.actions]

    def _is_collision(self, pos):
        x, y = pos[0], pos[1]
        sb = self.cfg['safety_buffer']
        # Hard-coded island check
        if (38 - sb <= x <= 90 + sb) and (43 - sb <= y <= 85 + sb): return True
        if np.sqrt((x-64)**2 + (y-17)**2) < (sb + 1.0): return True
        return x < 0 or x > 127 or y < 0 or y > 127

    def _simulate_trajectory(self, start_pos, start_vel, start_wind, action_seq):
        curr_pos, curr_vel = start_pos.copy(), start_vel.copy()
        
        for t, action_idx in enumerate(action_seq):
            # Wind Prediction
            theta = np.radians(self.mean_rotation * t)
            c, s = np.cos(theta), np.sin(theta)
            wind_t = np.array([start_wind[0]*c - start_wind[1]*s, start_wind[0]*s + start_wind[1]*c])
            
            # Physics Model
            a_dir = self.action_dirs[action_idx]
            eff = self._get_efficiency(a_dir, wind_t)
            v_target = eff * self.max_speed * a_dir
            curr_vel = self.inertia_factor * curr_vel + (1 - self.inertia_factor) * v_target
            
            v_disc = np.where(curr_vel < 0, np.ceil(curr_vel), np.floor(curr_vel)).astype(np.int32)
            curr_pos = np.clip(curr_pos + v_disc, [0, 0], [127, 127])
            
            # If any part of the trajectory crashes, the whole sequence is invalid
            if self._is_collision(curr_pos): return -5000.0 
            
        # Scoring: VMG vs Global Distance
        vmg = (curr_pos[1] - start_pos[1]) * self.cfg['vmg_weight']
        dist = -np.linalg.norm(curr_pos - self.goal_pos) * self.cfg['dist_weight']
        return vmg + dist

    def act(self, observation: np.ndarray) -> int:
        pos, vel, wind = observation[0:2], observation[2:4], observation[4:6]
        
        best_action, max_score = 0, -float('inf')
        
        for _ in range(self.cfg['num_samples']):
            # Random Shooting: Sample a sequence of actions
            seq = np.random.randint(0, 9, size=self.cfg['horizon'])
            score = self._simulate_trajectory(pos, vel, wind, seq)
            
            if score > max_score:
                max_score, best_action = score, seq[0]
                
        return int(best_action)

    def _get_efficiency(self, boat_dir, wind_dir):
        if np.all(boat_dir == 0): return 0.05
        wind_from = -wind_dir
        dot_p = np.clip(np.dot(wind_from, boat_dir), -1.0, 1.0)
        angle = np.arccos(dot_p)
        if angle < np.pi/4: return 0.05
        elif angle < np.pi/2: return 0.5 + 0.5 * (angle - np.pi/4) / (np.pi/4)
        elif angle < 3*np.pi/4: return 1.0
        else: return max(0.5, 1.0 - 0.5 * (angle - 3*np.pi/4) / (np.pi/4))