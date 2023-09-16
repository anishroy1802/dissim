import dissim
import matlab.engine
import os
import numpy as np
# Get the folder path where 'e4.m' is located
matlab_script_folder = r'C:\Users\hp\dissim\codes\algorithms\Simulated_Annealing\examples'

# Initialize the MATLAB Engine
eng = matlab.engine.start_matlab()
# Add the folder containing 'e4.m' to the MATLAB path
eng.addpath(matlab_script_folder)

def e4_wrapper(x):
    # Call the MATLAB e4 function and return the result as a Python float
    result = eng.e4(matlab.double(x), nargout=1)

    noise = np.random.normal(0, 0.3)  # Mean = 0, Standard Deviation = 0.3
    result_with_noise = float(result) + noise
    return result_with_noise

# Generate random noise 
# def generate_noise():
#     # Specify the mean (0) and standard deviation (0.3) for the normal distribution
#     mean = 0
#     std_dev = 0.3
#     # Generate random noise using numpy
#     noise = np.random.normal(mean, std_dev)
#     return noise

# Initial parameters for Simulated Annealing
initial_temperature = 100.0
cooling_rate = 0.5
num_iterations = 500
step_size = 0.2
domain_min = [0.0, 0.0]  # Minimum values for x[0] and x[1]
domain_max = [100.0, 100.0]  # Maximum values for x[0] and x[1]

#noise = generate_noise()
# Run the Simulated Annealing instance
sa_algorithm = dissim.SA(e4_wrapper, initial_temperature, cooling_rate, num_iterations, step_size, domain_min, domain_max)
sa_algorithm.run()

# Print details about the number of iterations
print("Number of iterations:", len(sa_algorithm.history))

# Print the optimal x and minimum value
result_x, result_min = sa_algorithm.history[-1]
print("Optimal x:", result_x)
print("Minimum value (with noise):", result_min)

# Plot the objective function variation over iterations
sa_algorithm.plot_objective_variation()

# Close the MATLAB Engine
eng.quit()