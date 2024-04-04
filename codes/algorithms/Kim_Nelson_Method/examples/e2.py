import dissim
import numpy as np

def func2(x):
  x1,x2,x3,x4 = x[0],x[1], x[2],x[3]
  return (x1+10*x2)**2 + 5* (x3-x4)**2 + (x2-2*x3)**4 + 10*(x1-x4)**4 + 1 +np.random.normal(0,30)

dom = [[1,7]]*4
step_size = [1,1,1,1]
opt  = dissim.KN(domain = dom, step_size= step_size,
                         func= func2,alpha=0.5, delta= 5, n_0  = 2, max_evals= 300)
a = opt.optimize()

#opt.print_function_values()
if a is not None:
    for ele in a:
        print("Element: ", opt.sol_space_dict[ele], "; Objective fn Value: ", -1* opt.X_i_bar[ele])
else:
    print("No solution")

