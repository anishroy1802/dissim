import dissim
import numpy as np
import matplotlib.pyplot as plt
import math as mt
import tracemalloc
import pandas as pd
import dask.dataframe as dd
import time
import random

def multinodal(x):
  return (np.sin(0.05*np.pi*x)**6)/2**(2*((x-10)/80)**2)

def func1(x0):
  x1,x2 = x0[0],x0[1]
  return -(multinodal(x1)+multinodal(x2))+np.random.normal(0,30)

#main()
dom = [[0,2], [0,2]]
step_size = [0.1, 0.2]
T= 100



optimizer  = dissim.SA(domain = dom, step_size= step_size, T = 100, max_evals= 1000,
                         func= func1, neigh_structure= 1, 
                         random_seed= 42, percent_improvement= 60)
optimizer.optimize()
optimizer.print_function_values()



# optimizer  = dissim.SA(domain = dom, step_size= step_size, T = 100, max_evals= 75,
#                          func= objective_function, neigh_structure= '2', 
#                          random_seed= 42)
# optimizer.optimize()
# optimizer.print_function_values()
