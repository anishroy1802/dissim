import dissim
import numpy as np

def func2(x):
  x1,x2,x3,x4 = x[0],x[1], x[2],x[3]
  return (x1+10*x2)**2 + 5* (x3-x4)**2 + (x2-2*x3)**4 + 10*(x1-x4)**4 
  + 1 +np.random.normal(0,30)


init = [2,2,2,2]
dom = [[1,7]]*4
step_size = [1,1,1,1]
# optimizer  = dissim.SA(domain = dom, step_size= step_size, T = 100, max_evals= 1000,
#                          func= func2, neigh_structure= 1, 
#                          random_seed= 42, percent_improvement= 40)
# optimizer.optimize()
# optimizer.print_function_values()

optimizer  = dissim.SA(domain = dom, step_size= step_size, T = 100, max_evals= 200,
                         func= func2, neigh_structure= 2, 
                         random_seed= 42)
optimizer.optimize()
optimizer.print_function_values()
