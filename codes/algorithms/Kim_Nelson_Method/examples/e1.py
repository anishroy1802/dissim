import dissim
import numpy as np

def multinodal(x):
  return (np.sin(0.05*np.pi*x)**6)/2**(2*((x-10)/80)**2)

def func1(x0, noise):
  x1,x2 = x0[0],x0[1]
  return -(multinodal(x1)+multinodal(x2))+noise
#main()
dom = [[0,2], [0,2]]
step_size = [0.5, 0.5]

optimizer  = dissim.KN(mu = 0, sigma = 30, domain = dom, step_size= step_size,
                         func= func1,alpha=0.5, delta= 5, n_0  = 2, max_evals= 1000)
a1 = optimizer.optimize()
#print("Solutions:")
# if a1 is not None:
#     for ele in a1:
#         print("Element: ", optimizer.sol_space_dict[ele], "; Objective fn Value: ", -1* optimizer.X_i_bar[ele])
# else:
#     print("No solution")

dom = [[0,2], [0,2]]
step_size = [0.5, 0.5]

optimizer  = dissim.KN(mu = 0, sigma = 30, domain = dom, step_size= step_size,
                         func= func1,alpha=0.5, delta= 5, n_0  = 2, max_evals= 1000, crn= True)
a1 = optimizer.optimize()
#print("Solutions:")
# if a1 is not None:
#     for ele in a1:
#         print("Element: ", optimizer.sol_space_dict[ele], "; Objective fn Value: ", -1* optimizer.X_i_bar[ele])
# else:
#     print("No solution")