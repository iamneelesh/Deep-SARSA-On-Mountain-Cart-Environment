import torch
import torch.nn.functional as F
from tqdm import tqdm

class DeepSARSAAgent:
    def __init__(self, q_network, target_network, memory, optimizer, gamma, epsilon):
        self.q_network = q_network
        self.target_network = target_network
        self.memory = memory
        self.optimizer = optimizer
        self.gamma = gamma
        self.epsilon = epsilon
    
    def select_action(self, state, epsilon):
        if torch.rand(1) < epsilon:
            return torch.randint(self.q_network.network[-1].out_features, (1, 1))
        else:
            with torch.no_grad():
                return self.q_network(state).argmax(dim=-1, keepdim=True)
    
    def update_q_network(self, batch_size):
        state_b, action_b, reward_b, done_b, next_state_b = self.memory.sample(batch_size)
        
        qsa_b = self.q_network(state_b).gather(1, action_b)
        next_action_b = self.select_action(next_state_b, self.epsilon)
        next_qsa_b = self.target_network(next_state_b).gather(1, next_action_b)
        target_b = reward_b + ~done_b * self.gamma * next_qsa_b
        
        loss = F.mse_loss(qsa_b, target_b)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
