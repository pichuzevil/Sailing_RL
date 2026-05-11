import numpy as np

def calculate_sailing_reward(obs, reward, terminated, info, prev_dist, curr_dist, gamma=0.995):
    """
    Computes a shaped reward considering boat physics and island proximity.
    """
    # 1. Base Environment Reward (+100 at goal, 0 otherwise)
    total_reward = reward 

    # 2. Potential-Based Distance Shaping
    # F = gamma * Phi(s') - Phi(s)
    dist_shaping = (gamma * curr_dist) - prev_dist
    total_reward -= dist_shaping * 0.1 

    # 3. Wind Efficiency (Point of Sail)
    wind_vec = obs[4:6] # Local wind
    vel_vec = obs[2:4]  # Boat velocity
    
    if np.linalg.norm(wind_vec) > 0 and np.linalg.norm(vel_vec) > 0:
        # Cosine similarity between wind and boat heading
        cos_theta = np.dot(wind_vec, vel_vec) / (np.linalg.norm(wind_vec) * np.linalg.norm(vel_vec))
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        
        # Penalize 'In Irons' (sailing into wind < 45 deg)
        if angle < np.pi / 4:
            total_reward -= 0.05
        # Bonus for Beam Reach (sailing perp to wind)
        elif np.pi/3 < angle < 2*np.pi/3:
            total_reward += 0.02

    # 4. Hard Crash Penalty (Increased to -50 to deter island magnets)
    if terminated and reward == 0:
        total_reward -= 50.0 

    # 5. Small step penalty to encourage speed
    total_reward -= 0.01

    return total_reward