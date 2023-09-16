import matplotlib.pyplot as plt
import math as mt
import numpy as np
import tracemalloc
import pandas as pd
import dask.dataframe as dd
import time
import random

"""
This code uses tracemalloc to trace memory allocation and measure peak memory usage. It defines two functions:

1. tracing_start():
   - Stops ongoing tracing, if any, and starts tracing memory allocation.
   - Helpful for detailed memory analysis, leak detection, and optimization.

2. tracing_mem():
   - Retrieves memory info, calculates peak usage in MB, and prints it.
   - Useful for monitoring peak memory during allocation tracing.

Overall, the code aids in understanding and optimizing memory usage.
"""

def tracing_start():
    """
    Starts tracing of memory allocation using tracemalloc.
    """
    tracemalloc.stop()
    print("nTracing Status : ", tracemalloc.is_tracing())
    tracemalloc.start()
    print("Tracing Status : ", tracemalloc.is_tracing())
def tracing_mem():
    """
    Prints the peak memory usage.
    """
    first_size, first_peak = tracemalloc.get_traced_memory()
    peak = first_peak/(1024*1024)
    print("Peak Size in MB - ", peak)


class SA():
    def __init__(self, objective_function, initial_temperature, cooling_rate, num_iterations, step_size, domain_min, domain_max):
        self.objective_function = objective_function
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.history = []

    def generate_neighbor(self, x):
        # Generate a random neighbor within the specified domain
        neighbor = x + self.step_size * np.random.randn(len(x))
        neighbor = np.clip(neighbor, self.domain_min, self.domain_max)  # Clip neighbor values to the domain
        return neighbor

    def metropolis_criterion(self, energy_difference, current_temperature):
        return energy_difference < 0 or np.random.rand() < np.exp(-energy_difference / current_temperature)

    def run(self):
        x = np.random.uniform(self.domain_min, self.domain_max)  # Initialize x randomly within the domain
        current_temperature = self.initial_temperature

        for iteration in range(self.num_iterations):
            for _ in range(10):  # Number of iterations at each temperature level
                neighbor = self.generate_neighbor(x)
                current_energy = self.objective_function(x)
                neighbor_energy = self.objective_function(neighbor)

                energy_difference = neighbor_energy - current_energy

                if self.metropolis_criterion(energy_difference, current_temperature):
                    x = neighbor

            self.history.append((x, self.objective_function(x)))
            current_temperature *= self.cooling_rate

    def plot_objective_variation(self):
        iterations = list(range(1, self.num_iterations + 1))
        objective_values = [item[1] for item in self.history]

        plt.plot(iterations, objective_values, marker='o')
        plt.xlabel("Iterations")
        plt.ylabel("Objective Function Value")
        plt.title("Objective Function Variation")
        plt.grid(True)
        plt.show()

#Define your noisy multivariable function here
# def objective_function(x):
#     noise = np.random.normal(scale=0.1)  # Add Gaussian noise with a standard deviation of 0.1
#     return x[0]**2 + x[1]**2 + noise

# Parameters
# initial_temperature = 100.0
# cooling_rate = 0.95
# num_iterations = 100
# step_size = 0.1
# domain_min = np.array([-1.0, 0.0])  # Minimum values for x[0] and x[1]
# domain_max = np.array([4.0, 10.0])  # Maximum values for x[0] and x[1]

# # Create and run the Simulated Annealing instance
# sa_algorithm = SA(objective_function, initial_temperature, cooling_rate, num_iterations, step_size, domain_min, domain_max)
# sa_algorithm.run()

# # Print details about number of iterations
# print("Number of iterations:", len(sa_algorithm.history))

# # Print optimal x and minimum value
# result_x, result_min = sa_algorithm.history[-1]
# print("Optimal x:", result_x)
# print("Minimum value (with noise):", result_min)

# # Plot the objective function variation over iterations
# sa_algorithm.plot_objective_variation()

"""
The code implements simulated annealing optimization with various parameters:

- initial_temperature: Controls initial exploration; higher values allow more search space coverage.
- cooling_rate: Governs convergence speed; lower values encourage thorough exploration.
- num_iterations: Defines optimization steps; higher values explore more options.
- step_size: Affects exploration granularity; smaller values fine-tune search.
- domain_min, domain_max: Restricts solution space; impacting feasible solutions.

Adjusting these parameters influences the balance between exploration and convergence, impacting optimization efficiency and solution quality.
"""
