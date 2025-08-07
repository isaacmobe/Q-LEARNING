import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import math

# Generate sample data for our regression problem
np.random.seed(42)
X = np.linspace(0, 10, 50)
true_slope = 2.5
true_intercept = 1.0
noise = np.random.normal(0, 1, 50)
y = true_slope * X + true_intercept + noise

# Our simple linear regression model
def linear_model(x, params):
    """Simple linear model: y = slope * x + intercept"""
    slope, intercept = params
    return slope * x + intercept

def calculate_error(params):
    """Calculate Mean Squared Error - this is what we want to minimize"""
    predictions = linear_model(X, params)
    mse = np.mean((y - predictions) ** 2)
    return mse

# =============================================================================
# GENETIC ALGORITHM (GA)
# =============================================================================
class GeneticAlgorithm:
    def __init__(self, population_size=20, mutation_rate=0.1, crossover_rate=0.8):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.best_fitness_history = []
        
    def create_individual(self):
        """Create a random individual (slope, intercept)"""
        # Initialize parameters in reasonable ranges
        slope = random.uniform(-5, 5)
        intercept = random.uniform(-5, 5)
        return [slope, intercept]
    
    def create_population(self):
        """Create initial population of random individuals"""
        return [self.create_individual() for _ in range(self.population_size)]
    
    def fitness(self, individual):
        """Fitness is inverse of error - lower error means higher fitness"""
        error = calculate_error(individual)
        return 1 / (1 + error)  # Add 1 to avoid division by zero
    
    def selection(self, population):
        """Tournament selection - pick best from random subset"""
        tournament_size = 3
        selected = []
        
        for _ in range(self.population_size):
            tournament = random.sample(population, tournament_size)
            winner = max(tournament, key=self.fitness)
            selected.append(winner)
        
        return selected
    
    def crossover(self, parent1, parent2):
        """Blend crossover - mix parameters from two parents"""
        if random.random() < self.crossover_rate:
            alpha = 0.5  # Blending factor
            child1 = [alpha * p1 + (1 - alpha) * p2 for p1, p2 in zip(parent1, parent2)]
            child2 = [(1 - alpha) * p1 + alpha * p2 for p1, p2 in zip(parent1, parent2)]
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
    
    def mutation(self, individual):
        """Add small random changes to parameters"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] += random.gauss(0, 0.5)  # Add Gaussian noise
        return mutated
    
    def optimize(self, generations=50):
        """Run the genetic algorithm"""
        population = self.create_population()
        
        for generation in range(generations):
            # Calculate fitness for all individuals
            fitness_scores = [self.fitness(ind) for ind in population]
            best_individual = population[np.argmax(fitness_scores)]
            best_error = calculate_error(best_individual)
            self.best_fitness_history.append(best_error)
            
            # Selection
            selected = self.selection(population)
            
            # Create next generation through crossover and mutation
            new_population = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            
            if generation % 10 == 0:
                print(f"GA Generation {generation}: Best Error = {best_error:.4f}")
        
        # Return best individual from final population
        fitness_scores = [self.fitness(ind) for ind in population]
        return population[np.argmax(fitness_scores)]

# =============================================================================
# PARTICLE SWARM OPTIMIZATION (PSO)
# =============================================================================
class ParticleSwarmOptimization:
    def __init__(self, num_particles=20, w=0.5, c1=2.0, c2=2.0):
        self.num_particles = num_particles
        self.w = w    # Inertia weight
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        self.best_fitness_history = []
    
    def optimize(self, iterations=100):
        """Run particle swarm optimization"""
        # Initialize particles with random positions and velocities
        particles = []
        for _ in range(self.num_particles):
            position = [random.uniform(-5, 5), random.uniform(-5, 5)]  # [slope, intercept]
            velocity = [random.uniform(-1, 1), random.uniform(-1, 1)]
            best_position = position.copy()
            particles.append({
                'position': position,
                'velocity': velocity,
                'best_position': best_position,
                'best_fitness': calculate_error(position)
            })
        
        # Find initial global best
        global_best_position = min(particles, key=lambda p: p['best_fitness'])['best_position'].copy()
        global_best_fitness = calculate_error(global_best_position)
        
        for iteration in range(iterations):
            for particle in particles:
                # Calculate current fitness
                current_fitness = calculate_error(particle['position'])
                
                # Update personal best
                if current_fitness < particle['best_fitness']:
                    particle['best_fitness'] = current_fitness
                    particle['best_position'] = particle['position'].copy()
                
                # Update global best
                if current_fitness < global_best_fitness:
                    global_best_fitness = current_fitness
                    global_best_position = particle['position'].copy()
                
                # Update velocity and position
                for i in range(len(particle['position'])):
                    r1, r2 = random.random(), random.random()
                    
                    # Velocity update equation
                    cognitive = self.c1 * r1 * (particle['best_position'][i] - particle['position'][i])
                    social = self.c2 * r2 * (global_best_position[i] - particle['position'][i])
                    
                    particle['velocity'][i] = (self.w * particle['velocity'][i] + 
                                             cognitive + social)
                    
                    # Position update
                    particle['position'][i] += particle['velocity'][i]
            
            self.best_fitness_history.append(global_best_fitness)
            
            if iteration % 10 == 0:
                print(f"PSO Iteration {iteration}: Best Error = {global_best_fitness:.4f}")
        
        return global_best_position

# =============================================================================
# SIMULATED ANNEALING (SA)
# =============================================================================
class SimulatedAnnealing:
    def __init__(self, initial_temp=100, cooling_rate=0.95, min_temp=0.01):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.best_fitness_history = []
    
    def get_neighbor(self, current_solution):
        """Generate a neighboring solution by adding small random changes"""
        neighbor = current_solution.copy()
        for i in range(len(neighbor)):
            neighbor[i] += random.gauss(0, 0.5)  # Add Gaussian noise
        return neighbor
    
    def acceptance_probability(self, current_energy, new_energy, temperature):
        """Calculate probability of accepting worse solution"""
        if new_energy < current_energy:
            return 1.0  # Always accept better solutions
        else:
            return math.exp(-(new_energy - current_energy) / temperature)
    
    def optimize(self, max_iterations=1000):
        """Run simulated annealing optimization"""
        # Start with random solution
        current_solution = [random.uniform(-5, 5), random.uniform(-5, 5)]
        current_energy = calculate_error(current_solution)
        
        best_solution = current_solution.copy()
        best_energy = current_energy
        
        temperature = self.initial_temp
        
        for iteration in range(max_iterations):
            # Generate neighbor solution
            neighbor = self.get_neighbor(current_solution)
            neighbor_energy = calculate_error(neighbor)
            
            # Decide whether to accept the neighbor
            if (neighbor_energy < current_energy or 
                random.random() < self.acceptance_probability(current_energy, neighbor_energy, temperature)):
                current_solution = neighbor
                current_energy = neighbor_energy
                
                # Update best solution if necessary
                if neighbor_energy < best_energy:
                    best_solution = neighbor.copy()
                    best_energy = neighbor_energy
            
            # Cool down the temperature
            if temperature > self.min_temp:
                temperature *= self.cooling_rate
            
            self.best_fitness_history.append(best_energy)
            
            if iteration % 100 == 0:
                print(f"SA Iteration {iteration}: Best Error = {best_energy:.4f}, Temp = {temperature:.4f}")
        
        return best_solution

# =============================================================================
# COMPARISON AND VISUALIZATION
# =============================================================================
def run_comparison():
    """Run all three optimization algorithms and compare results"""
    print("=== OPTIMIZATION COMPARISON ===")
    print(f"True parameters: slope = {true_slope}, intercept = {true_intercept}")
    print()
    
    # Run Genetic Algorithm
    print("Running Genetic Algorithm...")
    ga = GeneticAlgorithm()
    ga_best = ga.optimize(generations=50)
    ga_error = calculate_error(ga_best)
    print(f"GA Result: slope = {ga_best[0]:.3f}, intercept = {ga_best[1]:.3f}, Error = {ga_error:.4f}")
    print()
    
    # Run Particle Swarm Optimization
    print("Running Particle Swarm Optimization...")
    pso = ParticleSwarmOptimization()
    pso_best = pso.optimize(iterations=100)
    pso_error = calculate_error(pso_best)
    print(f"PSO Result: slope = {pso_best[0]:.3f}, intercept = {pso_best[1]:.3f}, Error = {pso_error:.4f}")
    print()
    
    # Run Simulated Annealing
    print("Running Simulated Annealing...")
    sa = SimulatedAnnealing()
    sa_best = sa.optimize(max_iterations=1000)
    sa_error = calculate_error(sa_best)
    print(f"SA Result: slope = {sa_best[0]:.3f}, intercept = {sa_best[1]:.3f}, Error = {sa_error:.4f}")
    print()
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Data and fitted lines
    plt.subplot(2, 2, 1)
    plt.scatter(X, y, alpha=0.6, label='Data Points')
    
    x_line = np.linspace(0, 10, 100)
    plt.plot(x_line, true_slope * x_line + true_intercept, 'k--', 
             label=f'True Line (slope={true_slope}, int={true_intercept})', linewidth=2)
    plt.plot(x_line, linear_model(x_line, ga_best), 'r-', 
             label=f'GA (slope={ga_best[0]:.2f}, int={ga_best[1]:.2f})')
    plt.plot(x_line, linear_model(x_line, pso_best), 'g-', 
             label=f'PSO (slope={pso_best[0]:.2f}, int={pso_best[1]:.2f})')
    plt.plot(x_line, linear_model(x_line, sa_best), 'b-', 
             label=f'SA (slope={sa_best[0]:.2f}, int={sa_best[1]:.2f})')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Fitted Lines Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Convergence comparison
    plt.subplot(2, 2, 2)
    plt.plot(ga.best_fitness_history, 'r-', label='Genetic Algorithm', linewidth=2)
    plt.plot(pso.best_fitness_history, 'g-', label='Particle Swarm', linewidth=2)
    plt.plot(sa.best_fitness_history, 'b-', label='Simulated Annealing', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Best Error (MSE)')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to better see differences
    
    # Plot 3: Error comparison bar chart
    plt.subplot(2, 2, 3)
    methods = ['Genetic\nAlgorithm', 'Particle\nSwarm', 'Simulated\nAnnealing']
    errors = [ga_error, pso_error, sa_error]
    colors = ['red', 'green', 'blue']
    
    bars = plt.bar(methods, errors, color=colors, alpha=0.7)
    plt.ylabel('Final Error (MSE)')
    plt.title('Final Error Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, error in zip(bars, errors):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{error:.3f}', ha='center', va='bottom')
    
    # Plot 4: Parameter accuracy
    plt.subplot(2, 2, 4)
    param_names = ['Slope', 'Intercept']
    x_pos = np.arange(len(param_names))
    width = 0.2
    
    true_params = [true_slope, true_intercept]
    ga_params = ga_best
    pso_params = pso_best
    sa_params = sa_best
    
    plt.bar(x_pos - 1.5*width, true_params, width, label='True', color='black', alpha=0.8)
    plt.bar(x_pos - 0.5*width, ga_params, width, label='GA', color='red', alpha=0.7)
    plt.bar(x_pos + 0.5*width, pso_params, width, label='PSO', color='green', alpha=0.7)
    plt.bar(x_pos + 1.5*width, sa_params, width, label='SA', color='blue', alpha=0.7)
    
    plt.xlabel('Parameters')
    plt.ylabel('Parameter Value')
    plt.title('Parameter Accuracy Comparison')
    plt.xticks(x_pos, param_names)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'GA': {'params': ga_best, 'error': ga_error},
        'PSO': {'params': pso_best, 'error': pso_error},
        'SA': {'params': sa_best, 'error': sa_error}
    }

# Run the comparison
if __name__ == "__main__":
    results = run_comparison()
    
    print("=== SUMMARY ===")
    for method, result in results.items():
        print(f"{method}: Parameters = {result['params']}, Error = {result['error']:.4f}")