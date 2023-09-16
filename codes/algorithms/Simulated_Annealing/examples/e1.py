import dissim
import numpy as np
import matplotlib.pyplot as plt
import math as mt
import tracemalloc
import pandas as pd
import dask.dataframe as dd
import time
import random
#Define your noisy multivariable function here
def objective_function(x):
    noise = np.random.normal(scale=0.1)  # Add Gaussian noise with a standard deviation of 0.1
    return x[0]**2 + x[1]**2 + noise

# Parameters
initial_temperature = 100.0
cooling_rate = 0.95
num_iterations = 100
step_size = 0.1
domain_min = np.array([-1.0, 0.0])  # Minimum values for x[0] and x[1]
domain_max = np.array([4.0, 10.0])  # Maximum values for x[0] and x[1]

# # Create and run the Simulated Annealing instance
sa_algorithm = dissim.SA(objective_function, initial_temperature, cooling_rate, num_iterations, step_size, domain_min, domain_max)
sa_algorithm.run()

# Print details about number of iterations
print("Number of iterations:", len(sa_algorithm.history))

# Print optimal x and minimum value
result_x, result_min = sa_algorithm.history[-1]
print("Optimal x:", result_x)
print("Minimum value (with noise):", result_min)

# Plot the objective function variation over iterations
sa_algorithm.plot_objective_variation()
