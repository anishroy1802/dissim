import dissim
import matlab.engine
import numpy as np

# Get the folder path where 'e6.m' is located
matlab_script_folder = r'C:\Users\hp\dissim\codes\algorithms\Stochastic_Ruler_Algorithm\examples'

# Initialize the MATLAB Engine
eng = matlab.engine.start_matlab()

# Add the folder containing 'e6.m' to the MATLAB path
eng.addpath(matlab_script_folder)

# Define the e6 function wrapper
def e6(x):
    # Extract x1 and x2 from the dictionary-like input
    x1 = x['x1']
    x2 = x['x2']
    
    # Call the MATLAB e6 function and return the result as a Python float
    result = eng.e6(float(x1), float(x2), nargout=1)

    noise = np.random.normal(0, 0.3)  # Mean = 0, Standard Deviation = 0.3
    result_with_noise = float(result) + noise
    return result_with_noise

dom = {'x1': [i for i in range(101)], 'x2': [i for i in range(101)]}
sr_userDef = dissim.stochastic_ruler(dom, 'user_defined', 100000, 'opt_sol', e6)
print(sr_userDef.optsol())

# Example usage with a dictionary-like input
#input_dict = {'x1': 10, 'x2': 10}
print(e6({'x1':10,'x2':10}))
