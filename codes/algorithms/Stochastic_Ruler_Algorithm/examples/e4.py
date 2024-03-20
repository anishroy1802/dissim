
import numpy as np
import dissim as ds

def objective_function(x):
    noise = np.random.normal(scale=0.1)  # Add Gaussian noise with a standard deviation of 0.1
    return 2*x['x1'] + x['x1']*2 + x['x1']*2 + noise

dom4 = dom = {"x1": [i for i in range(101)], "x2": [i for i in range(101)]}

sr_userDef4 = ds.stochastic_ruler(space = dom4, maxevals = 25, prob_type = 'opt_sol', func = objective_function, neigh_structure = 1 )
print(sr_userDef4.optsol())
# sr_userDef4.plot_minh_of_z()