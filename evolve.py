import numpy as np
import torch
import random
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from sklearn.decomposition import PCA
from model import MarioNet, mutate  # Import the model and mutation from model.py

# Set up environment and PCA
env = gym_super_mario_bros.make('SuperMarioBros-v3', apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
pca = PCA(n_components=50)

# Fitness evaluation function
def evaluate_fitness(info):
    return info['x_pos'] + info['score']

# Evolutionary loop
n_generations = 10
n_agents = 5
mutation_rate = 0.01
max_steps = 1000

# Initialize population of agents
population = [MarioNet(50, env.action_space.n) for _ in range(n_agents)]

for generation in range(n_generations):
    fitness_scores = []
    
    for agent in population:
        # Reset environment and run agent
        state = env.reset()
        state = pca.transform([state.flatten()])[0]
        total_reward = 0
        done = False
        
        for step in range(max_steps):
            action = random.choice(range(env.action_space.n))  # Random action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = pca.transform([next_state.flatten()])[0]
            state = next_state
            total_reward += reward
            if done:
                break
        
        fitness_scores.append(evaluate_fitness(info))
    
    # Select the top agents based on fitness
    top_agents_idx = np.argsort(fitness_scores)[-2:]
    top_agents = [population[i] for i in top_agents_idx]
    
    # Mutate and create a new population
    new_population = []
    for _ in range(n_agents):
        agent = random.choice(top_agents)
        new_agent = MarioNet(50, env.action_space.n)
        new_agent.load_state_dict(agent.state_dict())
        mutate(new_agent, mutation_rate)
        new_population.append(new_agent)
    
    population = new_population
    print(f"Generation {generation + 1} completed.")

env.close()
