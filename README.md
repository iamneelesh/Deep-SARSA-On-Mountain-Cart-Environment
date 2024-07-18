# Deep SARSA for Mountain Cart Environment

This project implements a Deep SARSA (State-Action-Reward-State-Action) algorithm to solve a custom Mountain Cart environment using PyTorch.

## Project Structure

The project is organized into the following Python files and a Jupyter notebook:

1. `environment.py`: Contains the custom Maze environment and utility functions
2. `model.py`: Defines the Q-network architecture
3. `agent.py`: Implements the Deep SARSA agent and training algorithm
4. `replay_memory.py`: Implements the replay buffer for experience replay
5. `main.py`: Orchestrates the training and evaluation process
6. `Deep_SARSA_Maze.ipynb`: Jupyter notebook containing the entire implementation and visualization

## File Descriptions

### environment.py
- Defines the `Maze` class, a custom gym environment
- Includes utility functions for visualization and data processing
- Contains functions for testing the agent and seeding the environment

### model.py
- Defines the `QNetwork` class, a neural network for Q-value approximation

### agent.py
- Implements the `DeepSARSAAgent` class
- Handles action selection, Q-network updates, and interaction with the environment

### replay_memory.py
- Defines the `ReplayMemory` class for experience replay
- Manages storage and sampling of transitions

### main.py
- Sets up the environment, agent, and training loop
- Runs the training process and visualizes results

### Deep_SARSA_Maze.ipynb
- Jupyter notebook containing the entire implementation
- Includes code cells for all components (environment, model, agent, etc.)
- Provides interactive visualizations and step-by-step execution
- Allows for easy experimentation with hyperparameters and immediate feedback

## Usage

### Using the Jupyter Notebook

1. Start Jupyter Notebook
2. Open `Deep_SARSA_on_Mountain_Cart_Environment.ipynb` in the Jupyter interface.

3. Run the cells in order to train the agent and visualize results.

## Results

The training process produces several visualizations:
- Training progress (loss and returns over time)
- Learned cost-to-go function for the Mountain Cart environment
- Optimal action map showing the best action for each state

These visualizations are available both through the Python scripts and in the Jupyter notebook.


## Acknowledgments

- OpenAI Gym for the environment structure
- PyTorch team for the deep learning framework
