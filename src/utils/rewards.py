import numpy as np
from sailing_physics import calculate_sailing_efficiency

def get_island_distance(x, y):
    """
    Calculates the exact Euclidean distance from (x, y) to the nearest point 
    on the hardcoded house-shaped island.
    """
    # 1. Distance to the Rectangle (Main Body)
    # Bounds: X[38, 90], Y[43, 85]
    dx = max(38 - x, 0, x - 90)
    dy = max(43 - y, 0, y - 85)
    dist_rect = np.sqrt(dx**2 + dy**2)

    # 2. Distance to the Triangle (The Southern Tip)
    # Apex: (64, 17), Base connects to rect at Y=43
    # This simplified distance covers the southern approach effectively
    dx_tri = max(38 - x, 0, x - 90)
    dy_tri = max(17 - y, 0, y - 43)
    dist_tri = np.sqrt(dx_tri**2 + dy_tri**2)

    return min(dist_rect, dist_tri)

def calculate_enhanced_reward(state, next_state, action, prev_action, reward, terminated, info, step_penalty=1.2, goal_pos=(64, 127)):
    # 1. Coordinate and Vector Extraction
    curr_pos = np.array(state[:2])
    next_pos = np.array(next_state[:2])
    wind_vec = np.array([next_state[4], next_state[5]])
    
    # 2. Time-Optimal Progress (Isochrone Logic)
    def get_sailing_dist(pos, wind):
        vec_to_goal = goal_pos - pos
        dist = np.linalg.norm(vec_to_goal)
        if dist < 1e-6: return 0
        
        unit_path = vec_to_goal / dist
        unit_wind = wind / (np.linalg.norm(wind) + 1e-6)
        eff = calculate_sailing_efficiency(unit_path, unit_wind)
        
        # Penalize distance by efficiency (lower eff = 'further' away)
        return dist / (eff + 0.1)

    s_dist_old = get_sailing_dist(curr_pos, wind_vec)
    s_dist_new = get_sailing_dist(next_pos, wind_vec)
    
    # Reward progress in 'Sailing Time' space
    shaped_r = (s_dist_old - s_dist_new) * 2.0
    
    # 3. GEOMETRIC ISLAND PENALTY (The "Force Field")
    # We use a 15-pixel "Rumble Strip" buffer around the exact island shape
    dist_to_land = get_island_distance(next_pos[0], next_pos[1])
    
    if dist_to_land < 4:
        # Exponential penalty: tiny at 15px, massive at 1px
        # This forces the agent to 'veer' away from the top-left corner
        proximity_penalty = (4 - dist_to_land) ** 1.5
        shaped_r -= proximity_penalty * 2

    # 4. Global Costs
    shaped_r -= step_penalty
    if action != prev_action:
        shaped_r -= -2.5 # Momentum penalty

    # 5. Terminal Constraints
    if terminated:
        if reward >= 100:
            shaped_r += 1000.0 # Goal bonus
        else:
            # Massive collision penalty to discourage corner-clipping
            shaped_r -= 1000.0 
            
    return shaped_r