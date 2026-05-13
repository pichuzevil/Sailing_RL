import numpy as np

def calculate_sailing_reward(obs, reward, terminated, info, prev_dist, curr_dist, gamma=0.99, step_penalty=0.5):
    total_reward = reward 

    # 1. Extreme VMG (The Speed Engine)
    pos_boat = obs[:2]
    vel_boat = obs[2:4]
    goal_pos = np.array([64, 127]) 
    vec_to_goal = goal_pos - pos_boat
    dist_to_goal = np.linalg.norm(vec_to_goal)
    
    if dist_to_goal > 1e-3:
        unit_goal = vec_to_goal / dist_to_goal
        vmg = np.dot(vel_boat, unit_goal)
        # We increase VMG weight to 6.0 to make speed 'everything'
        total_reward += vmg * 6.0 

    # 2. "Razor's Edge" Safety (Shrinking the buffer for speed)
    # To hit 45 steps, we must pass very close to the island.
    if 35 < pos_boat[1] < 85:
        dist_from_center = abs(pos_boat[0] - 64)
        if dist_from_center < 10: # Only 10 units wide now
            total_reward -= (10 - dist_from_center) * 1.5 # Very sharp 'pain' if too close

    # 3. Momentum Reward (Anti-Tack penalty)
    # Encourages long, fast runs rather than constant 'wobbling'
    # (This assumes the env provides 'last_action' or you can infer it)
    
    # 4. Point of Sail Optimization (Close Hauled)
    wind_vec = obs[4:6]
    if np.linalg.norm(wind_vec) > 0 and np.linalg.norm(vel_boat) > 0:
        cos_theta = np.dot(wind_vec, vel_boat) / (np.linalg.norm(wind_vec) * np.linalg.norm(vel_boat))
        angle = np.abs(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        
        # Penalize being 'In Irons' (0-35 degrees)
        if angle < 0.6: # ~35 degrees
            total_reward -= 0.5
        # MASSIVE bonus for the 'Racing' angle (40-50 degrees)
        elif 0.7 <= angle <= 0.9: 
            total_reward += 0.4

    # 5. Extreme Step Penalty (The 45-step Motivator)
    total_reward -= step_penalty 

    # 6. High Stakes Collision
    if info.get('collision', False) or (terminated and reward == 0):
        total_reward -= 200.0 

    return total_reward