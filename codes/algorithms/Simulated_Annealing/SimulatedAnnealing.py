"""" Package Imports:
In this section, we import several essential Python libraries that will be used throughout the code.
 - 'numpy' (imported as 'np') is a fundamental library for numerical operations, providing support for arrays and mathematical functions.
 - 'scipy.spatial.distance' imports the 'euclidean' function, which calculates the Euclidean distance between points.
 - 'tracemalloc' is used to monitor memory usage and allocation during code execution.
 - 'matplotlib.pyplot' (imported as 'plt') allows for creating various plots and visualizations.
 - 'math' provides a set of mathematical functions and constants for use in calculations."""

import numpy as np
from scipy.spatial.distance import euclidean
import tracemalloc
import matplotlib.pyplot as plt
import math

"""Memory Tracing:
 The 'tracing_start' function is defined to manage memory tracing. It begins by stopping any ongoing memory tracing using 'tracemalloc.stop()' and checking if tracing is currently active with 'tracemalloc.is_tracing()'. Afterward, it starts memory tracing with 'tracemalloc.start()' and checks the tracing status again. This function is invaluable for profiling memory usage, enabling developers to gain insights into memory allocation and deallocation patterns during code execution. It can help identify potential memory-related issues and optimize memory usage in the application."""

def tracing_start():
    tracemalloc.stop()
    print("Tracing Status: ", tracemalloc.is_tracing())
    tracemalloc.start()
    print("Tracing Status: ", tracemalloc.is_tracing())

class SA():
    def __init__(self, domain_min, domain_max, step_size, T, custom_H_function, nbd_structure, k= 300, percent_reduction=None):
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.step_size = step_size
        self.T = T
        self.k = k
        self.custom_H_function = custom_H_function
        self.solution_space = np.arange(domain_min, domain_max, step_size)
        self.n = len(self.solution_space)
        self.L = [i + 100 for i in range(0, k)]
        self.V = np.zeros(self.n)
        self.X_opt = []
        self.nbd_structure = nbd_structure
        self.function_values = []
        self.initial_function_value = None  # Store the initial function value
        self.percent_reduction = percent_reduction
        

    def create_neighborhood_and_transition_matrix(self):
        self.neighborhoods = {}
        if self.nbd_structure == "N1":
            for x in self.solution_space:
                N = [el for el in self.solution_space if el != x]
                self.neighborhoods[x] = N
        else:
            for i in range(self.n):
                neighbors = set()  # Create an empty set for neighbors of x[i].
                
                # Add the previous element (wrap-around to the last element if i is 0).
                neighbors.add(self.solution_space[(i - 1) % self.n])
                
                # Add the next element (wrap-around to the first element if i is n-1).
                neighbors.add(self.solution_space[(i + 1) % self.n])
                
                self.neighborhoods[self.solution_space[i]] = neighbors  # Store the neighbors of x[i] in the dictionary.

        if self.nbd_structure == "N1":
            R_prime = np.zeros((self.n, self.n))
            for i in range(self.n):
                for j in range(self.n):
                    R_prime[i, j] = euclidean([self.solution_space[i]], [self.solution_space[j]])

            D = np.sum(R_prime, axis=1)
            R = R_prime / D[:, np.newaxis]
        else:
            R = np.zeros((self.n, self.n))
            R_prime = np.zeros((self.n, self.n))

            for i in range(self.n):
                for j in range(self.n):
                    if self.solution_space[i] in self.neighborhoods[self.solution_space[j]]:
                        R[i][j] = 0.5

                    if R[i][j]!=0:
                        R_prime[i][j] = euclidean([self.solution_space[i]], [self.solution_space[j]])
            
        print("Transition Matrix:", R) 
        return R_prime, R

    
    def j_from_x(self, x):
        j = int(((x - self.domain_min) / (self.domain_max - self.domain_min)) * self.n)
        return j
    
    def optimize(self):
        R_prime, R = self.create_neighborhood_and_transition_matrix()

        X0 = np.random.choice(self.solution_space)
        X0 = self.solution_space[np.abs(self.solution_space - X0).argmin()]
        self.X_opt.append(X0)
        self.V[self.j_from_x(X0)] = 1

        for j in range(self.k):
            x_j = self.X_opt[j]
            N = self.neighborhoods[x_j]
            transition_probs = R[self.j_from_x(x_j)]
            transition_probs /= np.sum(transition_probs)
            z_j = np.random.choice(self.solution_space, p=transition_probs)
            z_j = self.solution_space[np.abs(self.solution_space - z_j).argmin()]
            fx = 0
            fz = 0

            for sim_iter in range(self.L[j]):
                fx += self.custom_H_function(x_j)
                fz += self.custom_H_function(z_j)

            fx = fx / self.L[j]
            fz = fz / self.L[j]
            G_xz = np.exp(-(fz - fx) / self.T)
            x_next = z_j if np.random.rand() <= G_xz else x_j
            self.V[self.j_from_x(x_next)] += 1
            D_x_j = np.sum(R_prime[self.j_from_x(x_j), :])
            D_x_next = np.sum(R_prime[self.j_from_x(x_next), :])
            x_opt_next = x_next if self.V[self.j_from_x(x_next)] / D_x_next > self.V[self.j_from_x(x_j)] / D_x_j else x_j
            self.X_opt.append(x_opt_next)

            # Calculate and store the function value for this iteration
            current_function_value = self.custom_H_function(x_opt_next)
            self.function_values.append(current_function_value)

            #if self.percent_reduction is not None:
            if self.initial_function_value is None:
                self.initial_function_value = current_function_value
            else:
                if self.percent_reduction is not None:
                    if ((self.initial_function_value - current_function_value) >= abs(self.initial_function_value) * (self.percent_reduction/100)):
                        print("Stopping criterion met (% reduction in function value). Stopping optimization.")
                        break
        
        # Plot the function value vs. iteration
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.function_values)), self.function_values, marker='o', linestyle='-')
        plt.xlabel('Iteration')
        plt.ylabel('Function Value')
        plt.title('Function Value vs. Iteration; Neighborhood Structure: ' + str(self.nbd_structure))
        plt.grid(True)
        plt.show()

    def print_function_values(self):
        print("iters:", len(self.function_values))
        #print("VALUES:")
        print("init:", self.function_values[0])
        print("final:", self.function_values[-1])
        print("% reduction:",  100*(self.function_values[0] -  self.function_values[-1])/abs(self.function_values[0]))
        #print(self.function_values)
"""
Main(), define H(x) and call optimizer to start the optimization for defined objective function
"""
def H(x):
    noise = np.random.normal(scale=0.1)
    return np.sin(2 * np.pi * x) + 0.5 * np.sin(4 * np.pi * x) + noise

"""
1. only computational budget option. set a default for this (300 func evals / sim reps). 
if specified and x% reduction is not specified, just run until budget is exhausted.
"""
# Example usage
optimizer = SA(domain_min=-1, domain_max=1, step_size=0.1, T=100, k=100, custom_H_function=H, nbd_structure="N1")
optimizer.optimize()
optimizer.print_function_values()

# Example usage
optimizer = SA(domain_min=-1, domain_max=1, step_size=0.1, T=100, k=100, custom_H_function=H, nbd_structure="N2")
optimizer.optimize()
optimizer.print_function_values()

"""
2. x% reduction specified and budget specified. 
if x% reduction achieved, terminate. 
else run until specified budget exhausted.
"""
# Example usage
optimizer = SA(domain_min=-1, domain_max=1, step_size=0.1, T=100, k=100, custom_H_function=H, nbd_structure="N1", percent_reduction=20)
optimizer.optimize()
optimizer.print_function_values()

# Example usage
optimizer = SA(domain_min=-1, domain_max=1, step_size=0.1, T=100, k=100, custom_H_function=H, nbd_structure="N2", percent_reduction=20)
optimizer.optimize()
optimizer.print_function_values()

"""
3. x% reduction specified and budget not specified. 
in this case, use the default the budget. if x% reduction achieved, terminate. 
else run until default budget exhausted. 
"""
# Example usage
optimizer = SA(domain_min=-1, domain_max=1, step_size=0.1, T=100,  custom_H_function=H, nbd_structure="N1", percent_reduction= 10)
optimizer.optimize()
optimizer.print_function_values()

# Example usage
optimizer = SA(domain_min=-1, domain_max=1, step_size=0.1, T=100,  custom_H_function=H, nbd_structure="N2", percent_reduction=10)
optimizer.optimize()
optimizer.print_function_values()