import dissim
import numpy as np

def multinodal(x):
  return (np.sin(0.05*np.pi*x)**6)/2**(2*((x-10)/80)**2)

def func1(x0):
  x1,x2 = x0[0],x0[1]
  return -(multinodal(x1)+multinodal(x2))+np.random.normal(0,30)

init = [2,2]
dom = [[1,7]]*2
step_size = [0.1,0.25]
func1AHA = dissim.AHA(func1,dom, step_size= step_size,max_evals= 500, 
                      percent_improvement=60, init_solution=[2,2],m = 40)
a = func1AHA.optimize()

func1AHA.print_function_values()

if a is not None:
  print(a[-1])
else:
  print("No solution")