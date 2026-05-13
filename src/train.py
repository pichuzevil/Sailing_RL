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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env_sailing import SailingEnv
from wind_scenarios import get_wind_scenario
from agents.dqn_agent import DQNAgent
from utils.replay_buffer import ReplayBuffer
from utils.rewards import calculate_sailing_reward
from utils.paths import get_dqn_save_path

def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN Sailing Agent")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_start", type=float, default=0.2, help="Start lower when refining weights")
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--target_update", type=int, default=10)
    parser.add_argument("--step_penalty", type=float, default=0.25)
    return parser.parse_args()

def get_distance(obs, goal_pos):
    return np.linalg.norm(obs[:2] - goal_pos)

def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # FOCUS on Scenario 3 (The Island) to break the 143-step barrier
    scenarios = ["training_1", "training_2", "training_3"]
    scenario_weights = [0.15, 0.15, 0.70] 
    
    weights_path = get_dqn_save_path()
    
    # FIX 1: Initialize ONCE and keep the loaded weights
    agent = DQNAgent(state_size=6, action_size=9, weights_path=weights_path)
    optimizer = optim.Adam(agent.policy_net.parameters(), lr=args.lr)
    
    memory = ReplayBuffer(capacity=50000, batch_size=args.batch_size, device=device)
    
    # FIX 2: Use the epsilon_start argument
    epsilon = args.epsilon_start
    best_metric = -float('inf') 
    success_window = deque(maxlen=100)
    reward_window = deque(maxlen=100)
    
    log_file = "src/agents/training_log.csv"
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "steps", "success"])

    print(f"🚀 Refinement started on {device}...")

    for ep in range(args.episodes):
        # FIX 3: Weighted scenario selection to master the hardest obstacle
        scene = np.random.choice(scenarios, p=scenario_weights)
        env = SailingEnv(**get_wind_scenario(scene))
        obs, info = env.reset()
        
        goal_pos = env.goal_position
        prev_dist = get_distance(obs, goal_pos)
        ep_shaped_reward = 0
        
        for step in range(500):
            # FIX 4: Use agent.act() to keep logic unified
            # We temporarily override agent.epsilon for the training loop
            agent.epsilon = epsilon
            action = agent.act(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            curr_dist = get_distance(next_obs, goal_pos)
            
            shaped_r = calculate_sailing_reward(
                obs=obs[:6], 
                reward=reward, 
                terminated=terminated, 
                info=info, 
                prev_dist=prev_dist, 
                curr_dist=curr_dist, 
                gamma=args.gamma,
                step_penalty=args.step_penalty
            )
            
            memory.push(obs[:6], action, shaped_r, next_obs[:6], terminated)
            
            if step % 4 == 0 and len(memory) > args.batch_size:
                states, actions, rewards, snext, dones = memory.sample()
                current_q = agent.policy_net(states).gather(1, actions)
                with torch.no_grad():
                    max_next_q = agent.target_net(snext).max(1)[0].unsqueeze(1)
                    target_q = rewards + (args.gamma * max_next_q * (1 - dones))
                
                loss = F.mse_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            obs = next_obs
            prev_dist = curr_dist
            ep_shaped_reward += shaped_r
            if terminated or truncated:
                break
        
        success = 1 if reward == 100 else 0
        success_window.append(success)
        reward_window.append(ep_shaped_reward)
        avg_success = np.mean(success_window)
        avg_reward = np.mean(reward_window)
        
        current_metric = (avg_success * 1000) + avg_reward
        
        if ep > 20 and current_metric > best_metric:
            best_metric = current_metric
            torch.save(agent.policy_net.state_dict(), weights_path)
            print(f" New Best! Metric: {best_metric:.2f} | Scene: {scene}")

        epsilon = max(0.05, epsilon * args.epsilon_decay)
        if ep % args.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep, ep_shaped_reward, step + 1, success])

    print(f"✅ Training finished.")

if __name__ == "__main__":
    train()