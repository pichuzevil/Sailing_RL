import sys
import os
import argparse
import random
import csv
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env_sailing import SailingEnv
from wind_scenarios import get_wind_scenario
from agents.dqn_agent import DQNAgent
from utils.replay_buffer import ReplayBuffer
from utils.rewards import calculate_enhanced_reward
from utils.paths import get_dqn_save_path

def parse_args():
    parser = argparse.ArgumentParser(description="Train an Optimized DQN Sailing Agent")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)      # Lower LR for stable refinement
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_start", type=float, default=0.3) # Increased for more island discovery
    parser.add_argument("--epsilon_decay", type=float, default=0.998)
    parser.add_argument("--target_update", type=int, default=10)
    parser.add_argument("--step_penalty", type=float, default=0.8)  # Lowered to prioritize safety over speed
    return parser.parse_args()

def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # FOCUS: Scenario 1 is where we crash, so we train on it 50% of the time
    scenarios = ["training_1", "training_2", "training_3"]
    scenario_weights = [0.33, 0.33, 0.34] 
    
    # UPGRADED: 11 Features (6 base + 4 navigation + 1 island radar)
    STATE_SIZE = 11 
    weights_path = get_dqn_save_path()
    
    # Initialize Agent with the 11-feature Dueling architecture
    agent = DQNAgent(state_size=STATE_SIZE, action_size=9, weights_path=weights_path)
    optimizer = optim.Adam(agent.policy_net.parameters(), lr=args.lr, weight_decay=1e-5)
    memory = ReplayBuffer(capacity=100000, batch_size=args.batch_size, device=device)
    
    epsilon = args.epsilon_start
    best_metric = -float('inf') 
    
    success_window = deque(maxlen=100)
    step_window = deque(maxlen=100)
    loss_window = deque(maxlen=100)
    
    log_file = "src/agents/training_log.csv"
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "steps", "success", "loss"])

    print(f"🚀 High-Performance Training (v11) started on {device}...")
    pbar = tqdm(range(args.episodes), desc="Training")

    for ep in pbar:
        scene = np.random.choice(scenarios, p=scenario_weights)
        env = SailingEnv(**get_wind_scenario(scene))
        obs, info = env.reset()
        goal_pos = env.goal_position 
        prev_action = 0
        ep_shaped_reward = 0.0
        ep_losses = []
        
        # Initial 11-feature vector
        state_vec = agent.preprocess_obs(obs, goal_pos=goal_pos)
        
        # Start at 0 penalty, ramp up to args.step_penalty by episode 1000
        if ep < 800:
            current_step_penalty = 0.1  # Low pressure to find the goal
        else:
            # Gradually increase penalty from 0.1 to 1.2
            progress = min(1.0, (ep - 800) / 700) 
            current_step_penalty = 0.1 + progress * (args.step_penalty - 0.1)
        
        for step in range(500):
            agent.epsilon = epsilon
            action = agent.act_from_vec(state_vec)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Get the next 11-feature vector
            next_state_vec = agent.preprocess_obs(next_obs, goal_pos=goal_pos)
            
            # Use the "Hardened" Reward function with higher collision penalties
            shaped_r = calculate_enhanced_reward(
                state=obs, 
                next_state=next_obs,
                action=action,
                prev_action=prev_action,
                reward=reward, 
                terminated=terminated, 
                info=info, 
                step_penalty=current_step_penalty,
                goal_pos=goal_pos 
            )
            
            memory.push(state_vec, action, shaped_r, next_state_vec, terminated)
            
            # Double DQN Optimization
            if len(memory) > args.batch_size:
                states, actions, rewards, snext, dones = memory.sample()
                current_q = agent.policy_net(states).gather(1, actions)
                
                with torch.no_grad():
                    # Action selection via policy net, evaluation via target net
                    next_actions = agent.policy_net(snext).argmax(1).unsqueeze(1)
                    max_next_q = agent.target_net(snext).gather(1, next_actions)
                    
                    rewards = rewards.view(-1, 1)
                    dones = dones.view(-1, 1)
                    target_q = rewards + (args.gamma * max_next_q * (1 - dones))
                
                loss = F.huber_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), 1.0)
                optimizer.step()
                ep_losses.append(loss.item())

            # Transition
            state_vec = next_state_vec
            obs = next_obs
            prev_action = action
            ep_shaped_reward += shaped_r
            
            if terminated or truncated:
                break
        
        # Metrics
        success = 1 if reward >= 100 else 0
        success_window.append(success)
        step_window.append(step + 1)
        avg_loss = np.mean(ep_losses) if ep_losses else 0
        loss_window.append(avg_loss)
        
        # Priority Metric: High success + Low steps
        # This will save only if the agent is safe AND fast
        current_metric = np.mean(success_window) * (500 - np.mean(step_window))
        
        if np.mean(success_window) > 0.80 and current_metric > best_metric:
            best_metric = current_metric
            torch.save(agent.policy_net.state_dict(), weights_path)
            tqdm.write(f"🌟 Ep {ep}: New Best! Metric: {best_metric:.1f} | Win: {np.mean(success_window)*100:.0f}% | Avg Steps: {np.mean(step_window):.1f}")

        pbar.set_postfix({
            "Steps": f"{np.mean(step_window):.1f}",
            "Win%": f"{np.mean(success_window)*100:.0f}%",
            "Loss": f"{avg_loss:.4f}",
            "Eps": f"{epsilon:.2f}"
        })

        epsilon = max(0.01, epsilon * args.epsilon_decay)
        if ep % args.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep, ep_shaped_reward, step + 1, success, avg_loss])

    print(f"✅ Training finished. Best model saved to {weights_path}")

if __name__ == "__main__":
    train()