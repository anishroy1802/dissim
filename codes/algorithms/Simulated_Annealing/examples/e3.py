import dissim
import numpy as np
import matplotlib.pyplot as plt
import math as mt
import tracemalloc
import pandas as pd
import dask.dataframe as dd
import time
import random

def objective_function(x):
    noise = np.random.normal(scale=0.1)  # Add Gaussian noise with a standard deviation of 0.1
    return 2*x[0] + x[0]**2 + x[1]**2 + x[2]*5+ noise

#main()
dom = [[0,2], [0,2], [0,2]]
step_size = [0.1, 0.2,0.5]
T= 100
k= 100

optimizer  = dissim.SA(domain = dom, step_size= step_size, T = 100, k = 25,
                         custom_H_function= objective_function, nbd_structure= 'N1', percent_reduction= 80)
optimizer.optimize()
#optimizer.print_function_values()