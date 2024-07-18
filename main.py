import torch
from torch.optim import AdamW
from environment import Maze, seed_everything, plot_stats, plot_cost_to_go, plot_max_q
from model import QNetwork
from agent import DeepSARSAAgent
from replay_memory import ReplayMemory

def main():
    # Hyperparameters
    EPISODES = 2500
    ALPHA = 0.001
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPSILON = 0.05
    MEMORY_CAPACITY = 1000000

    # Environment setup
    env = Maze(exploring_starts=True, shaped_rewards=True)
    seed_everything(env)
    state_dims = 2  # For Maze environment
    num_actions = env.action_space.n

    # Model setup
    q_network = QNetwork(state_dims, num_actions)
    target_q_network = QNetwork(state_dims, num_actions)
    target_q_network.load_state_dict(q_network.state_dict())
    target_q_network.eval()

    optimizer = AdamW(q_network.parameters(), lr=ALPHA)
    memory = ReplayMemory(MEMORY_CAPACITY)

    agent = DeepSARSAAgent(q_network, target_q_network, memory, optimizer, GAMMA, EPSILON)

    stats = {'MSE Loss': [], 'Returns': []}

    for episode in tqdm(range(1, EPISODES + 1)):
        state = env.reset()
        state = torch.tensor(state).unsqueeze(0).float()
        done = False
        ep_return = 0

        while not done:
            action = agent.select_action(state, EPSILON)
            next_state, reward, done, _ = env.step(action.item())
            next_state = torch.tensor(next_state).unsqueeze(0).float()
            reward = torch.tensor(reward).view(1, -1).float()
            done = torch.tensor(done).view(1, -1)

            memory.insert([state, action, reward, done, next_state])

            if memory.can_sample(BATCH_SIZE):
                loss = agent.update_q_network(BATCH_SIZE)
                stats['MSE Loss'].append(loss)

            state = next_state
            ep_return += reward.item()

        stats['Returns'].append(ep_return)

        if episode % 10 == 0:
            target_q_network.load_state_dict(q_network.state_dict())

    plot_stats(stats)
    plot_cost_to_go(env, q_network, xlabel='X position', ylabel='Y position')
    plot_max_q(env, q_network, xlabel='X position', ylabel='Y position',
               action_labels=['UP', 'RIGHT', 'DOWN', 'LEFT'])

if __name__ == "__main__":
    main()
