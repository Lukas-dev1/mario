import numpy as np
from sklearn.decomposition import PCA
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Setting up the environment
env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Collecting game states
n_steps = 1000
state_buffer = []

done = True
for step in range(n_steps):
    if done:
        state = env.reset()
    state, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated

    # Flatten the game frame (if it’s 2D, we make it 1D for PCA)
    flat_state = state.flatten()
    state_buffer.append(flat_state)

# Converting buffer to a numpy array
state_data = np.array(state_buffer)

# Applying PCA to reduce dimensions
n_components = 50  # You can adjust this to find the right balance
pca = PCA(n_components=n_components)
reduced_data = pca.fit_transform(state_data)

print(f"Original data shape: {state_data.shape}")
print(f"Reduced data shape: {reduced_data.shape}")

# You can use reduced_data for training your neural network
