import numpy as np
import dissim as ds

def func2(x):
  x1,x2,x3,x4 = x['x1'],x['x2'], x['x3'],x['x4']
  return (x1+10*x2)**2 + 5* (x3-x4)**2 + (x2-2*x3)**4 + 10*(x1-x4)**4 + 1 +np.random.normal(0,0.5)

dom3 = {'x1':[i for i in range(20)],'x2':[i for i in range(20)],'x3':
       [i for i in range(20)],'x4':[i for i in range(20)]}
sr_userDef3 = ds.stochastic_ruler(space = dom3, max_evals = 100, prob_type = 'opt_sol', func = func2, percent_improvement = 50, neigh_structure=2)
print(sr_userDef3.optsol())
# sr_userDef3.plot_minh_of_z()


# def func(x0):
#     x1, x2 = x0["x1"], x0["x2"]

#     def multinodal(x):
#         return (np.sin(0.05 * np.pi * x) ** 6) / 2 ** (2 * ((x - 10) / 80) ** 2)

#     return -(multinodal(x1) + multinodal(x2)) + np.random.normal(0, 0.3)


# dom = {"x1": [i for i in range(101)], "x2": [i for i in range(101)]}

# sr_userDef = stochastic_ruler(space=dom, maxevals=10, prob_type="opt_sol", func=func, neigh_structure=1)
# print(sr_userDef.optsol())
# #sr_userDef.plot_minh_of_z()