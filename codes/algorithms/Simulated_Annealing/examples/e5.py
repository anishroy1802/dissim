
import dissim
import numpy as np

def ex3(x):
    f = 0  # Default value
    if x[0] == 1:
        f = 0.3
    elif x[0] == 2:
        f = 0.7
    elif x[0] == 3:
        f = 0.9
    elif x[0] == 4:
        f = 0.5
    elif x[0] == 5:
        f = 1
    elif x[0] == 6:
        f = 1.4
    elif x[0] == 7:
        f = 0.7
    elif x[0] == 8:
        f = 0.8
    elif x[0] == 9:
        f = 0
    elif x[0] == 10:
        f = 0.6
    
    return f + np.random.uniform(-0.5, 0.5)

dom = [[1, 11]] * 1
step_size = [1] * len(dom)
T = 100
k = 100
optimizer = dissim.SA(dom=dom, step_size=step_size, T=100, k=100,
                      custom_H_function=ex3, nbd_structure='N1',
                      random_seed=42, percent_reduction=80)
optimizer.optimize()
optimizer.print_function_values()
