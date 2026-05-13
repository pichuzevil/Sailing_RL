import numpy as np

def calculate_sailing_reward(obs, reward, terminated, info, prev_dist, curr_dist, gamma=0.99, step_penalty=0.2):
    """
    Racing Spec Reward. 
    'step_penalty' is now passed from the command line.
    """
    total_reward = reward 

    # --- 1. VMG (Velocity Made Good) ---
    pos_boat = obs[:2]
    vel_boat = obs[2:4]
    goal_pos = np.array([64, 127]) 
    vec_to_goal = goal_pos - pos_boat
    dist_to_goal = np.linalg.norm(vec_to_goal)
    
    if dist_to_goal > 1e-3:
        unit_vec_to_goal = vec_to_goal / dist_to_goal
        vmg = np.dot(vel_boat, unit_vec_to_goal)
        total_reward += vmg * 3.0 

    # --- 2. Centerline Bias ---
    center_drift = abs(obs[0] - 64)
    total_reward -= (center_drift / 128.0) * 0.15

    # --- 3. Optimal Point of Sail ---
    wind_vec = obs[4:6]
    if np.linalg.norm(wind_vec) > 0 and np.linalg.norm(vel_boat) > 0:
        cos_theta = np.dot(wind_vec, vel_boat) / (np.linalg.norm(wind_vec) * np.linalg.norm(vel_boat))
        angle = np.abs(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        if angle < np.pi / 5:
            total_reward -= 0.3
        elif np.pi/5 <= angle <= np.pi/3.5:
            total_reward += 0.2

    # --- 4. Dynamic Step Penalty ---
    # Now uses the value passed from the train.py argument!
    total_reward -= step_penalty 

    # --- 5. Hard Crash Penalty ---
    if info.get('collision', False) or (terminated and reward == 0):
        total_reward -= 75.0

    return total_reward