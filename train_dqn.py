import numpy as np
import torch
import random
import matplotlib.pyplot as plt  # Import Matplotlib for image display
from collections import deque
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from sklearn.decomposition import PCA
from model import MarioNet  # Import from model.py

# Set up device (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up environment
env = gym_super_mario_bros.make('SuperMarioBros-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Buffer to collect initial game frames for PCA fitting
state_buffer = []
n_initial_steps = 500  # Reduced initial steps for PCA fitting

# Initial exploration to collect data for PCA
done = True
for step in range(n_initial_steps):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())

    # Collect game frames (flattened)
    flat_state = state.ravel()  # Ensure the state is flattened to 1D properly
    state_buffer.append(flat_state)

    # Display the game frame for debugging
    plt.imshow(state)  # Show the current game frame
    plt.axis('off')  # Turn off axis
    plt.title(f"Frame {step + 1}")  # Title with frame number
    plt.pause(0.1)  # Pause to display the frame for a moment

# Convert buffer to numpy array and fit PCA
state_data = np.array(state_buffer)
pca = PCA(n_components=50)
pca.fit(state_data)  # Fit PCA with the initial game frames

# Neural Network and DQN Setup
input_dim = 50  # PCA-reduced frame dimensions
output_dim = env.action_space.n  # Number of possible actions
model = MarioNet(input_dim, output_dim).to(device)  # Move model to the GPU
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64
replay_buffer = deque(maxlen=10000)

# Function to choose action (exploration vs exploitation)
def choose_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()  # Explore randomly
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # Move state to GPU
        q_values = model(state)
        return torch.argmax(q_values).item()  # Exploit learned knowledge

# Function to train the model on past experiences
def train_model():
    if len(replay_buffer) < batch_size:
        return
    
    mini_batch = random.sample(replay_buffer, batch_size)
    for state, action, reward, next_state, done in mini_batch:
        # Move tensors to GPU
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        reward = torch.FloatTensor([reward]).to(device)
        done = torch.FloatTensor([done]).to(device)
        
        # Compute target Q-value
        q_update = reward
        if not done:
            q_update += gamma * torch.max(model(next_state))
        
        # Get current Q-values
        q_values = model(state)
        q_target = q_values.clone().detach()
        q_target[0][action] = q_update

        # Optimize the model
        optimizer.zero_grad()
        loss = loss_fn(q_values, q_target)
        loss.backward()
        optimizer.step()

# Main training loop
n_episodes = 100  # Reduced number of episodes
max_steps = 100  # Reduced steps per episode

for episode in range(n_episodes):
    state = env.reset()
    state = np.array(state).ravel()  # Ensure state is flattened properly
    state = pca.transform([state])[0]  # Apply PCA to game frame
    total_reward = 0
    done = False

    for step in range(max_steps):
        if episode % 10 == 0:  # Render every 10 episodes
            env.render()

        # Choose action (random vs network)
        action = choose_action(state, epsilon)

        # Perform the action
        next_state, reward, done, info = env.step(action)

        # Flatten and apply PCA to the next state
        next_state = np.array(next_state).ravel()
        next_state_pca = pca.transform([next_state])[0]

        # Store the experience in the replay buffer
        replay_buffer.append((state, action, reward, next_state_pca, done))
        
        # Train the model using past experiences
        train_model()

        # Update state and total reward
        state = next_state_pca
        total_reward += reward

        if done:
            break

    # Decay epsilon to reduce randomness over time
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

env.close()
