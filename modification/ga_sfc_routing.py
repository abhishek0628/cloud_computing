# =========================
# Genetic Algorithm for SFC Routing Optimization in NTNs
# =========================

import random
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Parameters
# -------------------------
NUM_SATELLITES = 40        # Number of satellites
POP_SIZE = 30              # Population size
GENERATIONS = 80           # Iterations
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2
ROUTE_LENGTH = 5           # Each route passes through 5 satellites

# -------------------------
# Satellite resource data (simulated)
# -------------------------
delay = np.random.randint(10, 150, NUM_SATELLITES)
cpu = np.random.randint(1, 10, NUM_SATELLITES)
energy = np.random.randint(50, 200, NUM_SATELLITES)

# -------------------------
# Cost Function
# -------------------------
def cost_function(route):
    """Compute total cost for a route."""
    total_delay = sum(delay[i] for i in route)
    total_cpu = sum(cpu[i] for i in route)
    total_energy = sum(energy[i] for i in route)
    # Weighted combination of factors
    cost = 0.5 * total_delay + 0.3 * total_energy + 0.2 * total_cpu
    return cost

# -------------------------
# Initialize Population
# -------------------------
def initialize_population():
    return [random.sample(range(NUM_SATELLITES), ROUTE_LENGTH) for _ in range(POP_SIZE)]

# -------------------------
# Fitness Function
# -------------------------
def fitness(individual):
    return 1 / (1 + cost_function(individual))

# -------------------------
# Selection (Roulette Wheel)
# -------------------------
def selection(population):
    fitness_values = [fitness(ind) for ind in population]
    total_fit = sum(fitness_values)
    probs = [f / total_fit for f in fitness_values]
    parents = np.random.choice(range(POP_SIZE), size=2, p=probs)
    return population[parents[0]], population[parents[1]]

# -------------------------
# Crossover
# -------------------------
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, ROUTE_LENGTH - 2)
        child1 = parent1[:point] + [x for x in parent2 if x not in parent1[:point]]
        child2 = parent2[:point] + [x for x in parent1 if x not in parent2[:point]]
        return child1[:ROUTE_LENGTH], child2[:ROUTE_LENGTH]
    return parent1, parent2

# -------------------------
# Mutation
# -------------------------
def mutate(individual):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(ROUTE_LENGTH), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

# -------------------------
# Main GA Loop
# -------------------------
population = initialize_population()
best_costs = []

for gen in range(GENERATIONS):
    new_population = []
    for _ in range(POP_SIZE // 2):
        p1, p2 = selection(population)
        c1, c2 = crossover(p1, p2)
        new_population.append(mutate(c1))
        new_population.append(mutate(c2))

    population = new_population
    best = min(population, key=cost_function)
    best_costs.append(cost_function(best))

# -------------------------
# Results
# -------------------------
plt.figure(figsize=(8,5))
plt.plot(best_costs, linewidth=2)
plt.title("Genetic Algorithm Optimization for SFC Routing in NTNs")
plt.xlabel("Generation")
plt.ylabel("Total Cost")
plt.grid(True)

plt.savefig("cost_vs_generations.png")

plt.show()

print("Best Route Found:", best)
print("Minimum Cost:", round(cost_function(best), 2))

with open("results.txt", "w") as f:
    f.write(f"Best Route: {best}\n")
    f.write(f"Minimum Cost: {round(cost_function(best), 2)}\n")
print("\n✅ Results saved to results.txt")

