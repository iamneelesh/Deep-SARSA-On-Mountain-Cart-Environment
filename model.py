import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dims, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
    
    def forward(self, x):
        return self.network(x)
