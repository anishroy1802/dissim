import dissim
import matlab.engine
import os
import numpy as np
# Get the folder path where 'e4.m' is located
matlab_script_folder = r'C:\Users\hp\dissim\codes\algorithms\Adaptive_Hyperbox_Algorithm\examples'

# Initialize the MATLAB Engine
eng = matlab.engine.start_matlab()
# Add the folder containing 'e4.m' to the MATLAB path
eng.addpath(matlab_script_folder)


def e6_wrapper(x):
    # Call the MATLAB e4 function and return the result as a Python float
    result = eng.e6(matlab.double(x), nargout=1)

    noise = np.random.normal(0, 0.3)  # Mean = 0, Standard Deviation = 0.3
    result_with_noise = float(result) + noise
    return result_with_noise

init = [0,0]
dom = [[0,100],[0,100]]
func1AHA = dissim.AHA(e6_wrapper,dom)
a = func1AHA.AHAalgolocal(100,50,dom,init)
# print(b,c)
print(a[-1])
print(e6_wrapper(a[-1]))