import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.memory)

def get_epsilon(step, eps_start=1.0, eps_end=0.1, eps_decay=1e6):
    fraction = min(step / eps_decay, 1.0)
    return eps_start + fraction * (eps_end - eps_start)
