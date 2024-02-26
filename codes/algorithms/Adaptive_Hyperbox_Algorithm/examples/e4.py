import dissim
import numpy as np

def func10d(x):
    n = len(x)
    result = 0
    for i in range(n):
        result += x[i]**2
    result += (x[0] + 10*x[1])**2 + 5*(x[2] - x[3])**2 + (x[1] - 2*x[2])**4 + 10*(x[0] - x[3])**4 + 14*np.sum(x[:5]) + np.random.normal(0, 30)
    return result

dom10d = [[-100, 100]] * 10  # Define domain for 10 dimensions
init10d = [1] * 10  # Initial guess for 10 dimensions

func10d_AHA = dissim.AHA(func10d, dom10d, percent=60)
result = func10d_AHA.AHAalgolocal(100, dom10d, init10d, 4000)

print("Optimal solution:", result[-1])
print("Function value at optimal solution:", func10d(result[-1]))

func10d_AHA.plot_iterations()
