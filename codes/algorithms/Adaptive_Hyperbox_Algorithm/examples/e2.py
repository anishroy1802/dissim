import dissim
import numpy as np

def func2(x):
  x1,x2,x3,x4 = x[0],x[1], x[2],x[3]
  return (x1+10*x2)**2 + 5* (x3-x4)**2 + (x2-2*x3)**4 + 10*(x1-x4)**4 
  + 1 +np.random.normal(0,30)

init = [2,2,2,2]
dom = [[1,7]]*4
step_size = [0.1,0.25,0.5,0.5]
func1AHA = dissim.AHA(func2,dom, step_size= step_size,max_evals= 500, percent_improvement=60, init_solution=init,m = 40)
a = func1AHA.optimize()

func1AHA.print_function_values()


print(a[-1])