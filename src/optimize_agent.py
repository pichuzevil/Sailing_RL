import optuna
import numpy as np
from env_sailing import SailingEnv # Ensure this is in your path
from agents.agent_astar import MyAgent    # Ensure your updated agent is in your path

def objective(trial):
    # 1. Define Search Space
    config = {
        'max_nodes': trial.suggest_int('max_nodes', 20000, 100000),
        'safety_buffer': trial.suggest_float('safety_buffer', 0.5, 3.0),
        'replan_freq': trial.suggest_int('replan_freq', 3, 20),
        'h_factor': trial.suggest_float('h_factor', 0.4, 0.9),
        'bucketing_res': trial.suggest_float('bucketing_res', 0.5, 5.0),
        'vmg_weight': trial.suggest_float('vmg_weight', 0.5, 2.0)
    }

    # 2. Test across the 3 main training scenarios
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
            obs, reward, term, trunc, info = env.step(action)
            steps += 1
            
            if info.get('is_stuck', False):
                return 500 # Crash penalty
            
            done = term or trunc
            
        if not info.get('success', False):
            return 500 # Failure penalty

        total_steps += steps

    return total_steps / len(seeds)

# 3. Execute the study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print("🏆 Best Config found:", study.best_params)