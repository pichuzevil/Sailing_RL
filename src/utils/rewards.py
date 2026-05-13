import numpy as np

def calculate_enhanced_reward(state, next_state, action, prev_action, reward, terminated, info, step_penalty=1.2, goal_pos=(110, 10)):
    """
    High-performance reward shaping for sailing.
    
    Args:
        state: Current observation [x, y, vx, vy, wx, wy]
        next_state: Observation after action [x, y, vx, vy, wx, wy]
        action: Action taken (0-8)
        prev_action: Action taken in the previous step
        reward: Raw reward from the environment (e.g., 100 for goal)
        terminated: Boolean indicating if episode ended
        info: Environment info dict
        step_penalty: Cost of each move to encourage speed
        goal_pos: Coordinates of the fixed goal
    """
    # 1. High-Precision VMG (Velocity Made Good)
    # Vector from current boat position to the goal
    vec_to_goal = np.array([goal_pos[0] - next_state[0], goal_pos[1] - next_state[1]])
    dist_to_goal = np.linalg.norm(vec_to_goal)
    unit_vec_to_goal = vec_to_goal / (dist_to_goal + 1e-6)
    
    # Boat velocity vector
    vel_vec = np.array([next_state[2], next_state[3]])
    
    # Projection of velocity onto the goal vector
    vmg = np.dot(vel_vec, unit_vec_to_goal)
    
    # 2. Core Reward Logic
    # Multiply VMG to make it the primary signal (speed toward goal)
    shaped_r = vmg * 5.0 
    
    # Apply the step penalty to force the agent to find the 40-step path
    shaped_r -= step_penalty 
    
    # 3. Action Smoothness (Reduce Jitter)
    # Turning in sailing causes drag. Penalizing rapid switches maintains momentum.
    if action != prev_action:
        shaped_r -= 0.2
        
    # 4. Point of Sail Penalty ("In Irons")
    # Penalize pointing the boat directly into the wind (assuming wind is state[4:6])
    wind_vec = np.array([next_state[4], next_state[5]])
    if np.linalg.norm(wind_vec) > 0:
        unit_wind = wind_vec / np.linalg.norm(wind_vec)
        boat_dir_vec = vel_vec / (np.linalg.norm(vel_vec) + 1e-6)
        # Cosine similarity: 1.0 means sailing directly with wind, -1.0 means directly against
        cos_sim = np.dot(boat_dir_vec, unit_wind)
        if cos_sim < -0.8:  # Sailing too close to the wind direction
            shaped_r -= 0.5

    # 5. Termination Logic
    if terminated:
        if reward >= 100:
            # Huge bonus for hitting the actual goal
            shaped_r += 500.0 
        else:
            # Penalty for crashing or going out of bounds
            shaped_r -= 50.0

    return shaped_r