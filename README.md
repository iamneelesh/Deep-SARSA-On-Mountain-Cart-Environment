# Deep SARSA on Mountain Cart Environment

This project implements a Deep SARSA (State-Action-Reward-State-Action) algorithm to solve the Mountain Car problem using PyTorch and OpenAI Gym.

## Project Structure

The project is structured as a Jupyter Notebook with the following main sections:

1. **Environment Setup**
2. **Neural Network Definition**
3. **Replay Memory Implementation**
4. **Deep SARSA Algorithm**
5. **Training Loop**
6. **Visualization and Analysis**

## Requirements

- Python 3.7+
- Jupyter Notebook
- PyTorch
- OpenAI Gym
- NumPy
- Matplotlib
- tqdm

## Usage

1. Open the Jupyter Notebook
2. Run all cells in the notebook to train the agent and visualize results.

## Notebook Sections

### 1. Environment Setup
- Imports necessary libraries
- Creates and wraps the MountainCar-v0 environment

### 2. Neural Network Definition
- Defines the Q-network architecture
- Creates the target network

### 3. Replay Memory Implementation
- Implements a replay buffer for experience replay

### 4. Deep SARSA Algorithm
- Defines the deep_sarsa function
- Implements the SARSA update rule with neural network

### 5. Training Loop
- Runs the training process for a specified number of episodes
- Collects experience and updates the Q-network

### 6. Visualization and Analysis
- Plots training statistics (MSE loss and returns)
- Visualizes the learned cost-to-go function
- Displays the optimal action map

## Results

The notebook produces several visualizations:
- Training progress (loss and returns over time)
- Learned cost-to-go function for the Mountain Car problem
- Optimal action map showing the best action for each state

## Acknowledgments

- OpenAI Gym for the Mountain Car environment
- PyTorch team for the deep learning framework
