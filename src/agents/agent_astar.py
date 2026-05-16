import numpy as np
import heapq
from typing import Optional, Dict, Any

try:
    from agents.base_agent import BaseAgent 
except ImportError:
    from evaluator.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # 1. OPTUNA CONFIGURATION
        # If no config is provided, we use the best known baseline
        self.config = config or {
            'max_nodes': 50000,
            'safety_buffer': 2.0,
            'replan_freq': 8,
            'h_factor': 0.707,      # The VMG efficiency factor
            'bucketing_res': 2.0,    # Spatial precision
            'vmg_weight': 1.0       # Weight of vertical progress
        }

        # Environment Constants
        self.max_speed = 8.0
        self.inertia_factor = 0.3
        self.mean_rotation = 0.5
        self.goal_pos = np.array([64, 127])
        
        # Internal State
        self.world_map = None
        self.planned_path = []
        self.steps_since_plan = 0
        
        # Directions: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW, 8=Stay
        self.actions = [
            np.array([0, 1]), np.array([1, 1]), np.array([1, 0]), np.array([1, -1]),
            np.array([0, -1]), np.array([-1, -1]), np.array([-1, 0]), np.array([-1, 1]),
            np.array([0, 0])
        ]
        self.action_dirs = [a / np.linalg.norm(a) if np.any(a) else a for a in self.actions]

    def _get_efficiency(self, boat_dir, wind_dir):
        if np.all(boat_dir == 0): return 0.05
        wind_from = -wind_dir
        dot_p = np.clip(np.dot(wind_from, boat_dir), -1.0, 1.0)
        angle = np.arccos(dot_p)
        if angle < np.pi/4: return 0.05
        elif angle < np.pi/2: return 0.5 + 0.5 * (angle - np.pi/4) / (np.pi/4)
        elif angle < 3*np.pi/4: return 1.0
        else: return max(0.5, 1.0 - 0.5 * (angle - 3*np.pi/4) / (np.pi/4))

    def _predict_wind(self, initial_wind, t):
        theta = np.radians(self.mean_rotation * t)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([initial_wind[0]*c - initial_wind[1]*s, 
                         initial_wind[0]*s + initial_wind[1]*c])

    def _is_collision(self, pos):
        x, y = pos[0], pos[1]
        sb = self.config['safety_buffer']
        
        # Hard-coded Triangle Tip
        if np.sqrt((x - 64)**2 + (y - 17)**2) < (sb + 1.0):
            return True

        # Hard-coded Rectangle Body
        if (38 - sb <= x <= 90 + sb) and (43 - sb <= y <= 85 + sb):
            return True
            
        # Dynamic world_map check
        if self.world_map is not None:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    gx, gy = int(round(x + dx)), int(round(y + dy))
                    if 0 <= gx < 128 and 0 <= gy < 128:
                        if self.world_map[gy, gx] == 1: return True
        return x < 0 or x > 127 or y < 0 or y > 127

    def _crosses_goal(self, A, B, epsilon=1e-9):
        xA, yA, xB, yB, xP, yP = A[0], A[1], B[0], B[1], self.goal_pos[0], self.goal_pos[1]
        cross = (xB - xA) * (yP - yA) - (yB - yA) * (xP - xA)
        if abs(cross) > epsilon: return False
        return (min(xA, xB) - epsilon <= xP <= max(xA, xB) + epsilon and
                min(yA, yB) - epsilon <= yP <= max(yA, yB) + epsilon)

    def plan_path(self, start_pos, start_vel, start_wind):
        pq = [(0, 0, start_pos[0], start_pos[1], start_vel[0], start_vel[1], [])]
        visited = {} 
        best_h, best_path_so_far = float('inf'), [0]
        nodes_explored, max_nodes = 0, self.config['max_nodes']
        res = self.config['bucketing_res']

        while pq and nodes_explored < max_nodes:
            f, t, x, y, vx, vy, path = heapq.heappop(pq)
            nodes_explored += 1

            curr_pos = np.array([x, y])
            dist = np.linalg.norm(curr_pos - self.goal_pos)
            if dist < 1.5: return path
            if dist < best_h:
                best_h, best_path_so_far = dist, path

            state_key = (int(x // res), int(y // res), int(vx), int(vy))
            if state_key in visited and visited[state_key] <= t: continue
            visited[state_key] = t

            current_wind = self._predict_wind(start_wind, t)

            for i, a_dir in enumerate(self.action_dirs):
                eff = self._get_efficiency(a_dir, current_wind)
                v_next = self.inertia_factor * np.array([vx, vy]) + \
                         (1 - self.inertia_factor) * (eff * self.max_speed * a_dir)
                v_disc = np.where(v_next < 0, np.ceil(v_next), np.floor(v_next)).astype(np.int32)
                
                new_pos = curr_pos + v_disc
                
                is_safe = True
                for step in [0.25, 0.5, 0.75, 1.0]:
                    if self._is_collision(curr_pos + step * v_disc):
                        is_safe = False
                        break
                if not is_safe: continue

                # HEURISTIC: Using optimized factors
                y_remaining = 127 - new_pos[1]
                h = (y_remaining * self.config['vmg_weight']) / (self.max_speed * self.config['h_factor'])
                
                heapq.heappush(pq, (t + 1 + h, t + 1, new_pos[0], new_pos[1], v_disc[0], v_disc[1], path + [i]))

        return best_path_so_far

    def reset(self):
        self.planned_path, self.steps_since_plan, self.world_map = [], 0, None

    def act(self, observation: np.ndarray) -> int:
        pos, vel, wind = observation[0:2], observation[2:4], observation[4:6]
        if self.world_map is None and len(observation) > 100:
            self.world_map = observation[-16384:].reshape((128, 128))

        if not self.planned_path or self.steps_since_plan >= self.config['replan_freq']:
            self.planned_path = self.plan_path(pos, vel, wind)
            self.steps_since_plan = 0

        action = self.planned_path.pop(0) if self.planned_path else 0
        self.steps_since_plan += 1
        return int(action)