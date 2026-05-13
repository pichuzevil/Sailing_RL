import numpy as np

def calculate_sailing_reward(obs, reward, terminated, info, prev_dist, curr_dist, gamma=0.99, step_penalty=0.4):
    total_reward = reward 

    # --- 1. VMG (Keep the speed) ---
    pos_boat = obs[:2]
    vel_boat = obs[2:4]
    goal_pos = np.array([64, 127]) 
    vec_to_goal = goal_pos - pos_boat
    dist_to_goal = np.linalg.norm(vec_to_goal)
    
    if dist_to_goal > 1e-3:
        unit_vec_to_goal = vec_to_goal / dist_to_goal
        vmg = np.dot(vel_boat, unit_vec_to_goal)
        total_reward += vmg * 3.0 

    # --- 2. The "Safety Buffer" (Island Repulsion) ---
    # Scenario 3 Island is roughly between y=40 and y=80, centered at x=64.
    # If we are in that 'latitude', we punish the agent for being too close to the center.
    if 35 < pos_boat[1] < 85:
        dist_from_center = abs(pos_boat[0] - 64)
        if dist_from_center < 25: # If within 25 units of the island center
            # Penalty gets MUCH stronger the closer we get
            safety_penalty = (25 - dist_from_center) * 0.1
            total_reward -= safety_penalty

    # --- 3. Centerline Bias (Softened) ---
    # Only apply this when we are PAST the island (y > 85)
    if pos_boat[1] > 85:
        center_drift = abs(obs[0] - 64)
        total_reward -= (center_drift / 128.0) * 0.1

    # --- 4. Point of Sail ---
    wind_vec = obs[4:6]
    if np.linalg.norm(wind_vec) > 0 and np.linalg.norm(vel_boat) > 0:
        cos_theta = np.dot(wind_vec, vel_boat) / (np.linalg.norm(wind_vec) * np.linalg.norm(vel_boat))
        angle = np.abs(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        if angle < np.pi / 5:
            total_reward -= 0.3
        elif np.pi/5 <= angle <= np.pi/3.5:
            total_reward += 0.2

    # --- 5. Step Penalty ---
    total_reward -= step_penalty 

    # --- 6. The "Fear of God" (Restore high crash penalty) ---
    if info.get('collision', False) or (terminated and reward == 0):
        total_reward -= 150.0 # Bumped up to stop the 'gambling' behavior

    return total_reward