import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(file_path="src/agents/training_log.csv"):
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Log file not found. Run training first!")
        return

    # Window for smoothing (e.g., 50 episodes)
    window = 50
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 1. Plot Shaped Rewards 
    ax1.plot(data['episode'], data['reward'], alpha=0.2, color='blue', label='Raw Reward')
    ax1.plot(data['episode'], data['reward'].rolling(window=window).mean(), color='blue', label='Smoothed')
    ax1.set_title("Training Rewards (Shaped)")
    ax1.set_ylabel("Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Plot Success Rate 
    # We calculate the % of success over the last 'window' episodes
    success_rate = data['success'].rolling(window=window).mean() * 100
    ax2.plot(data['episode'], success_rate, color='green')
    ax2.set_title(f"Success Rate (Moving Average of {window} eps)")
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_xlabel("Episode")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("src/agents/learning_curve.png")
    plt.show()

if __name__ == "__main__":
    plot_learning_curve()