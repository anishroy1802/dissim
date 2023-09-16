import dissim
import numpy as np

def func2(x):
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    return (x1 + 10 * x2)**2 + 5 * (x3 - x4)**2 + (x2 - 2 * x3)**4 + 10 * (x1 - x4)**4 + 1 + np.random.normal(0, 30)

initial_temperature = 100.0
cooling_rate = 0.50
num_iterations = 500
step_size = 1.0
domain_min = np.array([-100.0, -100.0, -100.0, -100.0])  # Minimum values for x[0] x[1] x[2] x[3]
domain_max = np.array([100.0, 100.0, 100.0, 100.0])  # Maximum values for x[0] x[1] x[2] x[3]

# Create and run the Simulated Annealing instance
sa_algorithm = dissim.SA(func2, initial_temperature, cooling_rate, num_iterations, step_size, domain_min, domain_max)
sa_algorithm.run()

# Print details about the number of iterations
print("Number of iterations:", len(sa_algorithm.history))

# Print optimal x and minimum value
result_x, result_min = sa_algorithm.history[-1]
print("Optimal x:", result_x)
print("Minimum value (with noise):", result_min)

# Plot the objective function variation over iterations
sa_algorithm.plot_objective_variation()
