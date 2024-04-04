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

def e3_wrapper(x):
    try:
        # Convert x to a numpy array and then to a Python list of doubles
        x_double = np.array(x, dtype=float).tolist()
        
        # Call the MATLAB e4 function and return the result as a Python float
        result = eng.e3(matlab.double(x_double), nargout=1)

        noise = np.random.normal(0, 0.3)  # Mean = 0, Standard Deviation = 0.3
        result_with_noise = float(result) + noise
        return result_with_noise

    except matlab.engine.MatlabExecutionError as e:
        print(f'MATLAB Error: {e}')
        # Handle the error as needed
        return 0  # Default value or alternative handling


#main()
dom = [[0,2], [0,2]]
step_size = [0.5, 0.5]

optimizer  = dissim.KN(domain = dom, step_size= step_size,
                         func= e3_wrapper,alpha=0.5, delta= 5, n_0  = 2, max_evals= 300)
a1 = optimizer.optimize()
#print("Solutions:")
if a1 is not None:
    for ele in a1:
        print("Element: ", optimizer.sol_space_dict[ele], "; Objective fn Value: ", -1* optimizer.X_i_bar[ele])
else:
    print("No solution")