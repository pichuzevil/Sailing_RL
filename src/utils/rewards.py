import numpy as np

def calculate_sailing_reward(obs, reward, terminated, info, prev_dist, curr_dist, gamma=0.995):
    """
    Optimized for Speed and VMG (Velocity Made Good).
    Designed to cut step counts by rewarding high-speed progress.
    """
    # 1. Base Environment Reward (+100 at goal)
    total_reward = reward 

    # 2. VMG Calculation (Progress Velocity)
    # This rewards CLOSING the distance at high speed, not just being closer.
    pos_boat = obs[:2]
    vel_boat = obs[2:4]
    
    # Target is usually at (64, 127)
    goal_pos = np.array([64, 127]) 
    vec_to_goal = goal_pos - pos_boat
    dist_to_goal = np.linalg.norm(vec_to_goal)
    
    if dist_to_goal > 0:
        unit_vec_to_goal = vec_to_goal / dist_to_goal
        # VMG = Projection of velocity onto the goal direction
        vmg = np.dot(vel_boat, unit_vec_to_goal)
        # Higher multiplier (2.0) makes speed much more valuable than safe positioning
        total_reward += vmg * 2.0 

    # 3. Wind Efficiency & Optimal Tacking Angle
    # Upwind sailing is fastest at 'Close Hauled' (~45 degrees to wind).
    wind_vec = obs[4:6] 
    if np.linalg.norm(wind_vec) > 0 and np.linalg.norm(vel_boat) > 0:
        cos_theta = np.dot(wind_vec, vel_boat) / (np.linalg.norm(wind_vec) * np.linalg.norm(vel_boat))
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        
        # Heavy penalty for 'In Irons' (0-40 deg) to force the agent to keep its speed up
        if angle < np.pi / 4.5: 
            total_reward -= 0.2
        # 'Sweet Spot' Bonus: Close Hauled (40-50 deg)
        # This teaches the agent to point as high as possible without stalling.
        elif np.pi/4.5 <= angle <= np.pi/3.5:
            total_reward += 0.15
        # Beam Reach Bonus (60-120 deg) - good for speed but might add steps
        elif np.pi/3 < angle < 2*np.pi/3:
            total_reward += 0.05

    # 4. Increased Step Penalty (The 'Impatience' factor)
    # Moving from 0.05 to 0.15 forces the agent to take shorter paths.
    total_reward -= 0.15 

    # 5. Massive Crash Penalty
    if info.get('collision', False) or (terminated and reward == 0):
        total_reward -= 100.0 

    return total_reward