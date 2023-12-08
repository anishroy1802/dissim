import dissim
import numpy as np
import itertools


def func(x0):
  x1,x2 = x0[0],x0[1]
  def multinodal(x):
    return (np.sin(0.05*np.pi*x)**6)/2**(2*((x-10)/80)**2)

  return -(multinodal(x1)+multinodal(x2))+np.random.normal(0,0.3)

#main()

dom = [[0,2], [0,2]]
step_size = [0.1, 0.1]
T= 100
k= 100

# # Create and run the Simulated Annealing instance
optimizer = dissim.SA(dom = dom, step_size= step_size, 
T = 100, k = 100, custom_H_function= func, nbd_structure= 'N1', 
random_seed= 42, percent_reduction=60)
optimizer.optimize()
optimizer.print_function_values()

