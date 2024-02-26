import dissim
import matlab.engine
import os
import tracemalloc
import numpy as np
# Get the folder path where 'e4.m' is located

def tracing_start():
    tracemalloc.stop()
    print("nTracing Status : ", tracemalloc.is_tracing())
    tracemalloc.start()
    print("Tracing Status : ", tracemalloc.is_tracing())
def tracing_mem():
    first_size, first_peak = tracemalloc.get_traced_memory()
    peak = first_peak/(1024*1024)
    print("Peak Size in MB - ", peak)


matlab_script_folder = r'C:\Users\hp\dissim\codes\algorithms\Adaptive_Hyperbox_Algorithm\examples'

# Initialize the MATLAB Engine
eng = matlab.engine.start_matlab()
# Add the folder containing 'e4.m' to the MATLAB path
eng.addpath(matlab_script_folder)


def e6_wrapper(x):
    try:
        # Convert x to a numpy array and then to a Python list of doubles
        x_double = np.array(x, dtype=float).tolist()
        
        # Call the MATLAB e4 function and return the result as a Python float
        result = eng.e6(matlab.double(x_double), nargout=1)

        noise = np.random.normal(0, 0.3)  # Mean = 0, Standard Deviation = 0.3
        result_with_noise = float(result) + noise
        return result_with_noise

    except matlab.engine.MatlabExecutionError as e:
        print(f'MATLAB Error: {e}')
        # Handle the error as needed
        return 0  # Default value or alternative handling

init = [2,2]
dom = [[1,7]]*2
func1AHA = dissim.AHA(e6_wrapper,dom, percent=60)
a = func1AHA.AHAalgolocal(50,dom,init, 100)

func1AHA.plot_iterations()

