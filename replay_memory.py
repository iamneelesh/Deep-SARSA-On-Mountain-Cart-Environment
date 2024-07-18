import random
import torch

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def insert(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return map(torch.cat, zip(*batch))
    
    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size
