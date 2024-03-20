import numpy as np
import dissim as ds

def func2(x):
  x1,x2,x3,x4 = x['x1'],x['x2'], x['x3'],x['x4']
  return (x1+10*x2)**2 + 5* (x3-x4)**2 + (x2-2*x3)**4 + 10*(x1-x4)**4 + 1 +np.random.normal(0,30)

dom3 = {'x1':[i for i in range(20)],'x2':[i for i in range(20)],'x3':
       [i for i in range(20)],'x4':[i for i in range(20)]}
sr_userDef3 = ds.stochastic_ruler(space = dom3, maxevals = 100, prob_type = 'opt_sol', func = func2, percentReduction = 50)
print(sr_userDef3.optsol())
# sr_userDef3.plot_minh_of_z()


