import dissim
import numpy as np

def func2(x, noise):
  x1,x2,x3,x4 = x[0],x[1], x[2],x[3]
  return (x1+10*x2)**2 + 5* (x3-x4)**2 + (x2-2*x3)**4 + 10*(x1-x4)**4 + 1 + noise#+np.random.normal(0,30)

dom = [[1,3]]*4
step_size = [1,1,1,1]
opt  = dissim.KN(mu = 0, sigma= 30, domain = dom, step_size= step_size,
                         func= func2,alpha=0.5, delta= 5, n_0  = 2, max_evals= 5000, crn = True)
a = opt.optimize()

#opt.print_function_values()
if a is not None:
    for ele in a:
        print("Element: ", opt.sol_space_dict[ele], "; Objective fn Value: ", opt.X_i_bar[ele])
else:
    print("No solution")

