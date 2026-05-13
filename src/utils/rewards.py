import numpy as np

def calculate_sailing_reward(obs, reward, terminated, info, prev_dist, curr_dist, gamma=0.99):
    """
    Final Competition Version: Optimized for minimum steps and tight cornering.
    Combines VMG, Centerline Bias, and High-Urgency Penalties.
    """
    total_reward = reward # +100 for goal

    # --- 1. VMG (Velocity Made Good) ---
    # Rewards closure speed. This is the primary driver for faster times.
    pos_boat = obs[:2]
    vel_boat = obs[2:4]
    goal_pos = np.array([64, 127]) 
    
    vec_to_goal = goal_pos - pos_boat
    dist_to_goal = np.linalg.norm(vec_to_goal)
    
    if dist_to_goal > 1e-3:
        unit_vec_to_goal = vec_to_goal / dist_to_goal
        # VMG = Projection of velocity onto goal direction
        # $VMG = \vec{v} \cdot \hat{u}_{goal}$
        vmg = np.dot(vel_boat, unit_vec_to_goal)
        total_reward += vmg * 3.0 # High weight on speed

    # --- 2. Centerline Bias (The "Horseshoe Killer") ---
    # Prevents the agent from drifting to the far edges (like in Scenario 3).
    # Penalty increases as the boat moves away from the middle (x=64).
    center_drift = abs(obs[0] - 64)
    total_reward -= (center_drift / 128.0) * 0.15

    # --- 3. Optimal Point of Sail (Close Hauled) ---
    # Encourages the aerodynamic sweet spot for upwind speed.
    wind_vec = obs[4:6]
    if np.linalg.norm(wind_vec) > 0 and np.linalg.norm(vel_boat) > 0:
        cos_theta = np.dot(wind_vec, vel_boat) / (np.linalg.norm(wind_vec) * np.linalg.norm(vel_boat))
        angle = np.abs(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        
        # Penalty for 'In Irons' (Stalling)
        if angle < np.pi / 5:
            total_reward -= 0.3
        # Racing Bonus: Close Hauled (Fastest upwind angle)
        elif np.pi/5 <= angle <= np.pi/3.5:
            total_reward += 0.2

    # --- 4. Extreme Step Penalty ---
    # Tripled from the 'Safe' version. This creates massive urgency.
    # Taking 140 steps now costs -42 points. Taking 80 steps only costs -24.
    total_reward -= 0.3 

    # --- 5. Hard Crash Penalty ---
    # Lowered slightly to -75. If it's too high (-100), the agent won't 
    # take the risk of cutting the corner tightly.
    if info.get('collision', False) or (terminated and reward == 0):
        total_reward -= 75.0

    return total_reward