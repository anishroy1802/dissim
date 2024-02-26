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
    try:
        # Convert x to a numpy array and then to a Python list of doubles
        x_double = np.array(x, dtype=float).tolist()
        
        # Call the MATLAB e4 function and return the result as a Python float
        result = eng.e4(matlab.double(x_double), nargout=1)

        noise = np.random.normal(0, 0.3)  # Mean = 0, Standard Deviation = 0.3
        result_with_noise = float(result) + noise
        return result_with_noise

    except matlab.engine.MatlabExecutionError as e:
        print(f'MATLAB Error: {e}')
        # Handle the error as needed
        return 0  # Default value or alternative handling

# Generate random noise 
# def generate_noise():
#     # Specify the mean (0) and standard deviation (0.3) for the normal distribution
#     mean = 0
#     std_dev = 0.3
#     # Generate random noise using numpy
#     noise = np.random.normal(mean, std_dev)
#     return noise

# Initial parameters for Simulated Annealing
dom = [[1,20]]*2
step_size = [1]*len(dom)
T= 100
k= 100

optimizer  = dissim.SA(dom = dom, step_size= step_size, T = 100, k = 50,
                         custom_H_function= e4_wrapper, nbd_structure= 'N1', 
                         random_seed= 42, percent_reduction=40)

optimizer.optimize()
optimizer.print_function_values()

# Close the MATLAB Engine
eng.quit()