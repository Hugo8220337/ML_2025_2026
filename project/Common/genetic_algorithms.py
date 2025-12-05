import numpy as np
import random

def initialize_population(size, bounds):
    return np.array([
        [random.uniform(low, high) for low, high in bounds]
        for _ in range(size)
    ])

def tournament_selection(population, scores, tournament_size, maximize):
    pop_size = len(population)
    selected_indices = []
    
    for _ in range(pop_size):
        competitors_idx = np.random.randint(0, pop_size, size=tournament_size)
        competitors_scores = scores[competitors_idx]
        
        if maximize:
            winner_idx = competitors_idx[np.argmax(competitors_scores)]
        else:
            winner_idx = competitors_idx[np.argmin(competitors_scores)]
        
        selected_indices.append(winner_idx)
        
    return population[selected_indices]

def crossover_population(population, rate):
    next_generation = []
    pop_size, num_genes = population.shape
    
    for i in range(0, pop_size, 2):
        parent1 = population[i]
        parent2 = population[i+1] if i+1 < pop_size else population[0]
        
        if random.random() < rate and num_genes > 1:
            cut = random.randint(1, num_genes - 1)
            child1 = np.concatenate((parent1[:cut], parent2[cut:]))
            child2 = np.concatenate((parent2[:cut], parent1[cut:]))
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        
        next_generation.extend([child1, child2])
        
    return np.array(next_generation)[:pop_size]

def mutate_population(population, rate, bounds):
    pop_size, num_genes = population.shape
    for i in range(pop_size):
        for j in range(num_genes):
            if random.random() < rate:
                low, high = bounds[j]
                population[i][j] = random.uniform(low, high)
    return population

def check_early_stopping(current_best, global_best, maximize, min_delta):
    if maximize:
        return current_best > (global_best + min_delta)
    return current_best < (global_best - min_delta)

def run_genetic_algorithm(
    fitness_function,
    gene_bounds,
    population_size=50,
    generations=100,
    mutation_rate=0.1,
    crossover_rate=0.8,
    tournament_size=3,
    maximize=True,
    verbose=True,  
    patience=10,       
    min_delta=0.0001   
):
    population = initialize_population(population_size, gene_bounds)
    best_score = -float('inf') if maximize else float('inf')
    best_solution = None
    history = []
    no_improvement_counter = 0
    
    for generation in range(generations):
        scores = np.array([fitness_function(ind) for ind in population])
        
        current_best_idx = np.argmax(scores) if maximize else np.argmin(scores)
        current_best_score = scores[current_best_idx]
        
        if check_early_stopping(current_best_score, best_score, maximize, min_delta):
            best_score = current_best_score
            best_solution = population[current_best_idx]
            no_improvement_counter = 0 
        else:
            no_improvement_counter += 1
            
        history.append(best_score)
        
        if verbose and generation % 5 == 0:
            print(f"Gen {generation}: Best = {best_score:.4f} (No Improv: {no_improvement_counter})")

        if no_improvement_counter >= patience:
            if verbose:
                print(f"\nStopping early at generation {generation}.")
            break

        selected = tournament_selection(population, scores, tournament_size, maximize)
        offspring = crossover_population(selected, crossover_rate)
        population = mutate_population(offspring, mutation_rate, gene_bounds)

    return {
        "best_solution": best_solution,
        "best_score": best_score,
        "history": history,
        "generations_run": generation + 1
    }