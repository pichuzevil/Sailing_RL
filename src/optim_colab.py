import optuna
import numpy as np
import heapq
import os
from typing import Optional, Dict, Any

# 1. SETUP ENVIRONMENT
# If you are in Colab, ensure your files are in the path
import sys
sys.path.append('/content/Sailing_RL/src') 

from env_sailing import SailingEnv
from agents.agent_astar import MyAgent

# 2. THE OBJECTIVE FUNCTION
def objective(trial):
    # Search Space Constraints
    config = {
        'max_nodes': trial.suggest_int('max_nodes', 20000, 80000), # Lowered top end for speed
        'safety_buffer': trial.suggest_float('safety_buffer', 0.8, 2.5),
        'replan_freq': trial.suggest_int('replan_freq', 5, 12),
        'h_factor': trial.suggest_float('h_factor', 0.5, 0.8),
        'bucketing_res': trial.suggest_float('bucketing_res', 1.5, 4.0),
        'vmg_weight': trial.suggest_float('vmg_weight', 0.8, 1.4)
    }

    seeds = [1, 42, 1000]
    total_steps = 0
    
    for seed in seeds:
        env = SailingEnv()
        obs, _ = env.reset(seed=seed)
        agent = MyAgent(config=config)
        
        steps = 0
        done = False
        while not done and steps < 200:
            action = agent.act(obs)
            obs, _, term, trunc, info = env.step(action)
            steps += 1
            
            if info.get('is_stuck', False):
                return 500.0 # Crash penalty
            
            done = term or trunc
            
        if not info.get('success', False):
            return 500.0 # Timeout penalty

        total_steps += steps

    return total_steps / len(seeds)

# 3. EXECUTION WITH PARALLELISM
# We use n_jobs to run multiple trials simultaneously on the CPU cores.
# Even though we are not using the T4 for the A* math, this speeds up the study significantly.

# Create a persistent study (saves progress if Colab disconnects)
study_name = "sailing_optimization"
storage_name = f"sqlite:///{study_name}.db"

study = optuna.create_study(
    study_name=study_name, 
    storage=storage_name, 
    load_if_exists=True, 
    direction='minimize'
)

# n_jobs=-1 uses all available CPU cores
print("🚀 Starting Optimization on Colab...")
study.optimize(objective, n_trials=100, n_jobs=-1)

print("\n🏆 Best Config found:")
print(study.best_params)