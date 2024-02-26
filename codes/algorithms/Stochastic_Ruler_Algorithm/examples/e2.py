# Discrete valued function 

import numpy as np
import dissim as ds

def ex3(x):
    if x["x"] == 1:
        f = 0.3
    elif x["x"] == 2:
        f = 0.7
    elif x["x"] == 3:
        f = 0.9
    elif x["x"] == 4:
        f = 0.5
    elif x["x"] == 5:
        f = 1
    elif x["x"] == 6:
        f = 1.4
    elif x["x"] == 7:
        f = 0.7
    elif x["x"] == 8:
        f = 0.8
    elif x["x"] == 9:
        f = 0
    elif x["x"] == 10:
        f = 0.6
    return f + np.random.uniform(-0.5, 0.5)


dom2 = {"x": [i for i in range(1, 11)]}

sr_userdef2 = ds.stochastic_ruler(space=dom2, maxevals=300, prob_type="opt_sol", func=ex3, percentReduction=80, neigh_structure = 2 )
print(sr_userdef2.optsol())
sr_userdef2.plot_minh_of_z()