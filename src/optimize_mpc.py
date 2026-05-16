import optuna
from env_sailing import SailingEnv # Ensure path is correct
from agents.agent_mpc import MPCAgent

def objective(trial):
    # 1. Define MPC Search Space
    config = {
        'horizon': trial.suggest_int('horizon', 5, 25),
        'num_samples': trial.suggest_int('num_samples', 20, 100),
        'vmg_weight': trial.suggest_float('vmg_weight', 0.5, 2.5),
        'safety_buffer': trial.suggest_float('safety_buffer', 0.5, 3.0),
        'dist_weight': trial.suggest_float('dist_weight', 0.1, 1.5)
    }

    seeds = [1, 42, 1000] # Your 3 training scenarios
    total_steps = 0
    
    for seed in seeds:
        env = SailingEnv()
        obs, _ = env.reset(seed=seed)
        agent = MPCAgent(config=config)
        
        steps = 0
        done = False
        while not done and steps < 200:
            action = agent.act(obs)
            obs, _, term, trunc, info = env.step(action)
            steps += 1
            if info.get('is_stuck', False): return 500 # Penalty for crash
            done = term or trunc
            
        if not info.get('success', False): return 500 # Penalty for timeout
        total_steps += steps

    return total_steps / len(seeds)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("🏆 Best MPC Config:", study.best_params)