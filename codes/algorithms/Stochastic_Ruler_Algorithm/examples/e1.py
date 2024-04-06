# Multimodal function 

import numpy as np
import dissim as ds


def func(x0):
    x1, x2 = x0["x1"], x0["x2"]

    def multinodal(x):
        return (np.sin(0.05 * np.pi * x) ** 6) / 2 ** (2 * ((x - 10) / 80) ** 2)

    return -(multinodal(x1) + multinodal(x2)) + np.random.normal(0, 0.3)


dom = {"x1": [i for i in range(101)], "x2": [i for i in range(101)]}

sr_userDef = ds.stochastic_ruler(space=dom, max_evals=20, prob_type="opt_sol", func=func, neigh_structure=1)
print(sr_userDef.optsol())