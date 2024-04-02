import dissim
import numpy as np

def objective_function(x):
    noise = np.random.normal(scale=0.1)  # Add Gaussian noise with a standard deviation of 0.1
    return -1*(2*x[0] + x[0]**2 + x[1]**2 + noise) # Minimisation problem hence -1 

#main()
dom = [[0,2], [0,2]]
step_size = [0.5, 0.5]

optimizer  = dissim.KN(domain = dom, step_size= step_size,
                         func= objective_function,alpha=0.5, delta= 5, n_0  = 2, max_evals= 300)
a1 = optimizer.optimize()
#print("Solutions:")
if a1 is not None:
    for ele in a1:
        print("Element: ", optimizer.sol_space_dict[ele], "; Objective fn Value: ", -1* optimizer.X_i_bar[ele])
else:
    print("No solution")