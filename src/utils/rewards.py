import numpy as np

def calculate_enhanced_reward(state, next_state, action, prev_action, reward, terminated, info, step_penalty=1.2, goal_pos=(64, 127)):
    """
    Hardened Reward Shaping for Sailing Navigation.
    Optimized for the 'House' island geometry and No-Go Zone physics.
    """
    # 1. State Extraction
    curr_pos = np.array(state[:2])
    next_pos = np.array(next_state[:2])
    vel_vec = np.array([next_state[2], next_state[3]])
    wind_vec = np.array([next_state[4], next_state[5]])
    
    # 2. High-Precision VMG (Velocity Made Good)
    vec_to_goal = np.array([goal_pos[0] - next_pos[0], goal_pos[1] - next_pos[1]])
    dist_to_goal = np.linalg.norm(vec_to_goal)
    unit_vec_to_goal = vec_to_goal / (dist_to_goal + 1e-6)
    
    # Projection of velocity onto the goal vector
    vmg = np.dot(vel_vec, unit_vec_to_goal)
    shaped_r = vmg * 5.0 
    
    # 3. Time Pressure (Step Penalty)
    shaped_r -= step_penalty 

    # 4. Physics-Aware "In Irons" Penalty
    # According to sailing_physics.py, < 45 degrees to wind is the No-Go Zone.
    if np.linalg.norm(wind_vec) > 0 and np.linalg.norm(vel_vec) > 0:
        unit_wind_to = wind_vec / np.linalg.norm(wind_vec)
        unit_boat_dir = vel_vec / np.linalg.norm(vel_vec)
        
        # cos_sim -1.0 means pointing directly INTO the wind (North wind vs North boat)
        cos_sim = np.dot(unit_boat_dir, unit_wind_to)
        
        # 45 degrees corresponds to cos_sim of -0.707
        if cos_sim < -0.707: 
            # Efficiency is only 0.05 here; we punish the agent for wasting time
            shaped_r -= 2.0 

    # 5. Static Island "Radar" Penalty (Proactive Avoidance)
    # The 'House' island center is at [64, 51]. 
    island_center = np.array([64, 51])
    dist_to_island = np.linalg.norm(next_pos - island_center)
    
    # If within 25 pixels of the island center, apply a 'grazing' penalty.
    # This prevents the agent from trying to cut the corner at [64, 17].
    if dist_to_island < 30:
        shaped_r -= 0.5

    # 6. Action Smoothness
    if action != prev_action:
        shaped_r -= 1.0 # Slightly increased to favor long tacks

    # 7. Termination Logic (CRITICAL FIX)
    if terminated:
        if reward >= 100:
            # Reached Goal
            shaped_r += 1000.0 
        else:
            # CRASHED or STUCK: This must be huge to outweigh the step_penalty!
            # If the agent crashes, it loses more than it could ever save in time.
            shaped_r -= 500.0

    return shaped_r