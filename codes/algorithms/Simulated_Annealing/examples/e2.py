import dissim
import numpy as np
def multinodal(x):
  return (np.sin(0.05*np.pi*x)**6)/2**(2*((x-10)/80)**2)

def func1(x0):
  x1,x2 = x0[0],x0[1]
  return -(multinodal(x1)+multinodal(x2)) +np.random.normal(0,0.3)


initial_temperature = 100.0
cooling_rate = 0.5
num_iterations = 500
step_size = 0.2
domain_min = np.array([0.0, 0.0])  # Minimum values for x[0] and x[1]
domain_max = np.array([100.0, 100.0])  # Maximum values for x[0] and x[1]

# # Create and run the Simulated Annealing instance
sa_algorithm = dissim.SA(func1, initial_temperature, cooling_rate, num_iterations, step_size, domain_min, domain_max)
sa_algorithm.run()

# Print details about number of iterations
print("Number of iterations:", len(sa_algorithm.history))

# Print optimal x and minimum value
result_x, result_min = sa_algorithm.history[-1]
print("Optimal x:", result_x)
print("Minimum value (with noise):", result_min)

# Plot the objective function variation over iterations
sa_algorithm.plot_objective_variation()