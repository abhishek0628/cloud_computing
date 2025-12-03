# models/ga_sfc.py
import numpy as np

class GASFC:
    def __init__(self, num_vnfs, num_nodes, pop_size=50, generations=100, mutation_rate=0.1):
        self.num_vnfs = num_vnfs
        self.num_nodes = num_nodes
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = np.random.randint(0, num_nodes, size=(pop_size, num_vnfs))

    def fitness(self, deployment):
        # Example cost function: sum of node indices
        return 1.0 / (1 + np.sum(deployment))

    def selection(self, fitness_scores):
        idx = np.random.choice(self.pop_size, size=self.pop_size, p=fitness_scores/fitness_scores.sum())
        return self.population[idx]

    def crossover(self, parent1, parent2):
        point = np.random.randint(1, self.num_vnfs)
        child = np.concatenate([parent1[:point], parent2[point:]])
        return child

    def mutate(self, individual):
        for i in range(self.num_vnfs):
            if np.random.rand() < self.mutation_rate:
                individual[i] = np.random.randint(0, self.num_nodes)
        return individual

    def run(self):
        best_cost = float('inf')
        best_deployment = None
        cost_history = []

        for gen in range(self.generations):
            fitness_scores = np.array([self.fitness(ind) for ind in self.population])
            parents = self.selection(fitness_scores)

            next_population = []
            for i in range(0, self.pop_size, 2):
                p1, p2 = parents[i], parents[i+1]
                c1 = self.crossover(p1, p2)
                c2 = self.crossover(p2, p1)
                next_population.append(self.mutate(c1))
                next_population.append(self.mutate(c2))

            self.population = np.array(next_population)
            gen_costs = [1.0 / f for f in fitness_scores]
            gen_best_idx = np.argmin(gen_costs)
            if gen_costs[gen_best_idx] < best_cost:
                best_cost = gen_costs[gen_best_idx]
                best_deployment = self.population[gen_best_idx]
            cost_history.append(best_cost)
            print(f"Generation {gen} | Best Cost: {best_cost:.4f}")

        return best_deployment, best_cost, cost_history
