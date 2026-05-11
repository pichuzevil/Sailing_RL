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

# Setup pathing for local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env_sailing import SailingEnv
from wind_scenarios import get_wind_scenario
from agents.dqn_agent import DQNAgent
from utils.replay_buffer import ReplayBuffer
from utils.rewards import calculate_sailing_reward

def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN Sailing Agent")
    parser.add_argument("--episodes", type=int, default=5000, help="Total episodes")
    parser.add_argument("--batch_size", type=int, default=256, help="Larger batch for T4 GPU efficiency")
    parser.add_argument("--lr", type=float, default=5e-5, help="Lower LR for fine-tuning weights")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--epsilon_decay", type=float, default=0.9995, help="Slow decay to master Scenario 1")
    parser.add_argument("--target_update", type=int, default=10, help="Target sync frequency")
    return parser.parse_args()

def get_distance(obs, goal_pos):
    """Euclidean distance to goal."""
    return np.linalg.norm(obs[:2] - goal_pos)

def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scenarios = ["training_1", "training_2", "training_3"]
    
    # Path for Google Drive
    save_path = "/content/drive/MyDrive/dqn_weights.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Initialize Agent and Optimizer
    agent = DQNAgent(state_size=6, action_size=9)
    optimizer = optim.Adam(agent.policy_net.parameters(), lr=args.lr)
    
    # Use Normalized Replay Buffer
    memory = ReplayBuffer(capacity=50000, batch_size=args.batch_size, device=device)
    
    # Tracking for "Save Best" logic
    epsilon = 1.0
    best_metric = -float('inf') 
    success_window = deque(maxlen=100)
    reward_window = deque(maxlen=100)
    
    log_file = "src/agents/training_log.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "steps", "success"])

    print(f"🚀 Starting training on {device}...")

    for ep in range(args.episodes):
        scene = random.choice(scenarios)
        env = SailingEnv(**get_wind_scenario(scene))
        obs, info = env.reset()
        
        goal_pos = env.goal_position
        prev_dist = get_distance(obs, goal_pos)
        ep_shaped_reward = 0
        
        for step in range(500): #
            # Epsilon-greedy selection
            if random.random() < epsilon:
                action = random.randint(0, 8)
            else:
                # IMPORTANT: Normalize live inference ONLY here
                state_t = torch.FloatTensor(obs[:6]).unsqueeze(0).to(device)
                state_t[:, 0:2] /= 128.0 
                with torch.no_grad():
                    action = agent.policy_net(state_t).argmax().item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            curr_dist = get_distance(next_obs, goal_pos)
            
            # Calculate Reward via Utility
            shaped_r = calculate_sailing_reward(obs, reward, terminated, info, prev_dist, curr_dist, args.gamma)
            
            # Push RAW obs (Buffer handles internal normalization)
            memory.push(obs[:6], action, shaped_r, next_obs[:6], terminated)
            
            # Optimization step every 4 steps for speed
            if step % 4 == 0 and len(memory) > args.batch_size:
                states, actions, rewards, snext, dones = memory.sample()

                # NO manual normalization here; ReplayBuffer.sample() did it!
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
        
        # Calculate performance metrics
        success = 1 if reward == 100 else 0
        success_window.append(success)
        reward_window.append(ep_shaped_reward)
        
        avg_success = np.mean(success_window)
        avg_reward = np.mean(reward_window)
        
        # --- COMPOSITE SAVE BEST LOGIC ---
        # Metric = (SuccessRate * 1000) + AvgReward 
        # This prioritizes reaching the goal, then falling back to distance/efficiency
        current_metric = (avg_success * 1000) + avg_reward
        
        if ep > 50 and current_metric > best_metric:
            best_metric = current_metric
            torch.save(agent.policy_net.state_dict(), save_path)
            print(f"🌟 Saved New Best (Succ: {avg_success:.2f}, Rew: {avg_reward:.2f})")

        # Update exploration and target network
        epsilon = max(0.05, epsilon * args.epsilon_decay)
        if ep % args.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
        # Logging to CSV
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep, ep_shaped_reward, step + 1, success])

        if ep % 50 == 0:
            print(f"Ep {ep:4d} | Metric: {current_metric:7.2f} | Eps: {epsilon:.3f} | Scene: {scene}")

    print(f"✅ Training finished. Weights saved to Google Drive.")

if __name__ == "__main__":
    train()