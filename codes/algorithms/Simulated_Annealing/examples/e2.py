import dissim
import numpy as np
import matplotlib.pyplot as plt
import math as mt
import tracemalloc
import pandas as pd
import dask.dataframe as dd
import time
import random
#Define your noisy multivariable function here
def facility_loc(x):
  #normal demand

  X1,Y1 = x[0],x[1] 
  X2,Y2 = x[2],x[3] 
  X3,Y3 = x[4],x[5] 
  avg_dist_daywise = []
  T0 = 30
  n = 6
  for t in range(T0):
      total_day = 0                  ##### total distance travelled by people
      ###### now finding nearest facility and saving total distance 
      #travelled in each entry of data
      for i in range(n):
          for j in range(n):
              demand=-1
              while(demand<0):    
                  demand = np.random.normal(180, 30, size=1)[0]
              total_day += demand*min(abs(X1-i)+abs(Y1-j) ,
                                      abs(X2-i)+abs(Y2-j),abs(X3-i)+abs(Y3-j) ) 
              ### total distance from i,j th location to nearest facility
      avg_dist_daywise.append(total_day/(n*n))    
  return sum(avg_dist_daywise)/T0


dom = [[1,4]]*6
step_size = [1]*len(dom)
T= 100
k= 100


optimizer  = dissim.SA(domain = dom, step_size= step_size, T = 100, k = 50,
                         custom_H_function= facility_loc, nbd_structure= 'N1', percent_reduction=40)
optimizer.optimize()
#optimizer.print_function_values()
