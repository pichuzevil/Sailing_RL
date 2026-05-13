import numpy as np

def calculate_enhanced_reward(state, next_state, action, prev_action, goal_pos=(110, 10)):
    # 1. High-Precision VMG
    # Vector from boat to goal
    vec_to_goal = np.array([goal_pos[0] - next_state[0], goal_pos[1] - next_state[1]])
    dist_to_goal = np.linalg.norm(vec_to_goal)
    unit_vec_to_goal = vec_to_goal / (dist_to_goal + 1e-6)
    
    # Boat velocity vector
    vel_vec = np.array([next_state[2], next_state[3]])
    
    # Projection of velocity onto goal vector
    vmg = np.dot(vel_vec, unit_vec_to_goal)
    
    reward = vmg * 5.0  # Strong incentive for speed toward goal
    
    # 2. Step Penalty (To hit the 40-step target)
    reward -= 1.5 
    
    # 3. Action Smoothness (Reduce Jitter)
    if action != prev_action:
        reward -= 0.2  # Small penalty for switching directions
        
    # 4. Critical Failures
    if next_state[0] <= 0 or next_state[0] >= 128 or next_state[1] <= 0 or next_state[1] >= 128:
        reward -= 50.0 # Out of bounds
        
    return reward