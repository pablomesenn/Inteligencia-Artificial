import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

# GA parameters 
POP_SIZE = 30
GENS = 50
MUT_RATE = 0.1

# Environment setup
env = gym.make("CartPole-v1", render_mode="human")

# Each individual will have 4 weights -> action = sign(obs · w)
def policy(obs, weights):
    return 0 if np.dot(obs, weights) < 0 else 1

# Evaluate fitness of an individual
def evaluate(weights):
    total_reward = 0
    # Obs -> [cart position, cart velocity, pole angle, pole angular velocity]
    # _ -> extra info, not used here
    obs, _ = env.reset()
    done = False
    while not done:
        action = policy(obs, weights)
        obs, reward, terminated, truncated, _ = env.step(action)
        # For visualization of the info
        print(obs, reward, terminated, truncated)
        total_reward += reward
        done = terminated or truncated
    return total_reward

# Crossover strategies

def single_point_crossover(p1, p2):
    point = np.random.randint(1, len(p1))
    child = np.concatenate([p1[:point], p2[point:]])
    return child

def two_point_crossover(p1, p2):
    point1 = np.random.randint(1, len(p1)-1)
    point2 = np.random.randint(point1+1, len(p1))
    child = np.concatenate([p1[:point1], p2[point1:point2], p1[point2:]])
    return child

def uniform_crossover(p1, p2):
    mask = np.random.randint(0, 2, size=len(p1))
    child = np.where(mask, p1, p2)
    return child

# Population initialization
population = [np.random.uniform(-1, 1, size=4) for _ in range(POP_SIZE)]

# Lists to record fitness progress
avg_fitness_per_gen = []
max_fitness_per_gen = []

for gen in range(GENS):
    # Evaluate fitness
    fitness = [evaluate(ind) for ind in population]

    avg_fitness = np.mean(fitness)
    max_fitness = np.max(fitness)
    avg_fitness_per_gen.append(avg_fitness)
    max_fitness_per_gen.append(max_fitness)

    # Select the top 20% of the generation
    sorted_pop = [x for _, x in sorted(zip(fitness, population), key=lambda x: x[0], reverse=True)]
    survivors = sorted_pop[:POP_SIZE // 5]

    # Reproduction
    new_pop = []
    for _ in range(POP_SIZE):
        p1, p2 = random.sample(survivors, 2)
        # Crossover
        child = uniform_crossover(p1, p2) #* Can change crossover method here calling other of the above functions
        # Mutation
        if np.random.rand() < MUT_RATE:
            child += np.random.uniform(-0.5, 0.5, size=child.shape)
        new_pop.append(child)
    
    population = new_pop

env.close()

# Just to indicate the end of training and everything went fine
print("Training complete.")

plt.figure(figsize=(10,5))
plt.plot(range(GENS), avg_fitness_per_gen, label="Average Fitness")
plt.plot(range(GENS), max_fitness_per_gen, label="Max Fitness")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Fitness Progression Over Generations")
plt.legend()
plt.grid(True)
plt.show()