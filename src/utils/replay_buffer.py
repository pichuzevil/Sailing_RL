import numpy as np
import torch
import random
from collections import deque

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples with automatic normalization."""

    def __init__(self, capacity, batch_size, device):
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.device = device

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        """Randomly sample a batch and normalize coordinates."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states, actions, rewards, next_states, dones = zip(*experiences)

        # Convert to tensors and move to device
        states_t = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions_t = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards_t = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states_t = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones_t = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

        # Automatic Normalization: Scale coordinates (indices 0 and 1) from [0, 128] to [0, 1]
        #
        states_t[:, 0:2] /= 128.0
        next_states_t[:, 0:2] /= 128.0

        return (states_t, actions_t, rewards_t, next_states_t, dones_t)

    def __len__(self):
        return len(self.memory)