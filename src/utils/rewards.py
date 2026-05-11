import numpy as np

def calculate_sailing_reward(obs, reward, terminated, info, prev_dist, curr_dist, gamma=0.995):
    """
    Advanced reward shaping designed to solve Scenario 1 and prevent island crashes.
    """
    # 1. Base Environment Reward (+100 for goal)
    total_reward = reward 

    # 2. Aggressive Distance Shaping (Prevents the 'Stay Still' bug)
    # Reward is strictly based on how many units closer we got.
    # If we move away or stay still, this becomes 0 or negative.
    dist_improvement = prev_dist - curr_dist
    total_reward += dist_improvement * 0.5  # Strong 'pull' North

    # 3. Wind Efficiency (Point of Sail)
    wind_vec = obs[4:6] 
    vel_vec = obs[2:4]  
    
    if np.linalg.norm(wind_vec) > 0 and np.linalg.norm(vel_vec) > 0:
        cos_theta = np.dot(wind_vec, vel_vec) / (np.linalg.norm(wind_vec) * np.linalg.norm(vel_vec))
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        
        # Penalty for 'In Irons' (stalling into wind)
        if angle < np.pi / 4: 
            total_reward -= 0.1  # Increased penalty
        # Bonus for Beam Reach (speed optimization)
        elif np.pi/3 < angle < 2*np.pi/3:
            total_reward += 0.05

    # 4. Island Proximity Buffer (Preventing the 'Blown into side' issue)
    # If the environment info provides 'crash_risk' or you check the map:
    # We punish the agent for being too close to the island BEFORE it hits.
    #
    if info.get('collision', False) or (terminated and reward == 0):
        total_reward -= 100.0  # Massive 'fear' of the island
    
    # 5. Step Penalty (The 'Time is Money' motivator)
    # This must be larger than any 'sitting still' bonus.
    total_reward -= 0.05 

    return total_reward