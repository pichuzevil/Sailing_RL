import numpy as np

def calculate_sailing_reward(obs, reward, terminated, info, prev_dist, curr_dist, gamma=0.99, step_penalty=0.8):
    total_reward = reward # +100 for goal

    pos_boat = obs[:2]
    vel_boat = obs[2:4]
    goal_pos = np.array([64, 127]) 
    dist_to_goal = np.linalg.norm(goal_pos - pos_boat)

    # 1. Hyper-VMG (The Primary Driver)
    # We increase the weight to 10.0 to make speed the absolute priority.
    if dist_to_goal > 1e-3:
        vmg = np.dot(vel_boat, (goal_pos - pos_boat) / dist_to_goal)
        total_reward += vmg * 10.0 

    # 2. The "Short-Cut" Buffer
    # We shrink the safety zone to 8 units. 
    # This allows the boat to shave the corner of the island.
    if 35 < pos_boat[1] < 85:
        dist_from_island_center = abs(pos_boat[0] - 64)
        if dist_from_island_center < 8: 
            total_reward -= (8 - dist_from_island_center) * 2.0

    # 3. Boundary Anti-Hugging Penalty
    # If the boat is too far left (x < 20) or right (x > 108), we punish it.
    # This stops the 'Horseshoe' path seen in your image.
    if pos_boat[0] < 25 or pos_boat[0] > 103:
        total_reward -= 0.5

    # 4. Point of Sail (Aerodynamics)
    wind_vec = obs[4:6]
    if np.linalg.norm(wind_vec) > 0 and np.linalg.norm(vel_boat) > 0:
        cos_theta = np.dot(wind_vec, vel_boat) / (np.linalg.norm(wind_vec) * np.linalg.norm(vel_boat))
        angle = np.abs(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        
        # Penalize 'In Irons' (0-40 degrees)
        if angle < 0.7: 
            total_reward -= 0.6
        # Bonus for 'Close Hauled' Racing Angle
        elif 0.7 <= angle <= 0.9:
            total_reward += 0.5

    # 5. Extreme Step Penalty
    # At -0.8, the agent loses points so fast it HAS to find the shortcut.
    total_reward -= step_penalty 

    # 6. High-Stakes Crash
    if info.get('collision', False) or (terminated and reward == 0):
        total_reward -= 500.0 

    return total_reward