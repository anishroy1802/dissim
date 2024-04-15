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
import itertools
import random
import pandas as pd
"""Memory Tracing:
 The 'tracing_start' function is defined to manage memory tracing. It begins by stopping any ongoing memory tracing using 'tracemalloc.stop()' and checking if tracing is currently active with 'tracemalloc.is_tracing()'. Afterward, it starts memory tracing with 'tracemalloc.start()' and checks the tracing status again. This function is invaluable for profiling memory usage, enabling developers to gain insights into memory allocation and deallocation patterns during code execution. It can help identify potential memory-related issues and optimize memory usage in the application."""

def tracing_start():
    tracemalloc.stop()
    print("Tracing Status: ", tracemalloc.is_tracing())
    tracemalloc.start()
    print("Tracing Status: ", tracemalloc.is_tracing())

# Implement soln with max_evals as overall simulation budget with a default 5 reps per solution
#k = simulation budget
class SA():

    def __init__(self, domain, step_size, T, func, neigh_structure = 1, max_evals= 300,  print_solutions = True, init_solution=None, random_seed=None, percent_improvement=None):

        self.domain = domain #n*2 matrix
        self.dimensions = len(self.domain)
        print("Number of dimensions:", self.dimensions)
        self.step_size = step_size #n*1 matrix
        self.T = T
        self.max_evals = max_evals
        self.func = func
        # self.L = [i + 200 for i in range(0, k)]
        self.init_solution = init_solution
        self.random_seed = random_seed
        self.X_opt = []
        self.fx_opt = []
        self.neigh_structure = neigh_structure
        self.function_values = []
        self.initial_function_value = None  # Store the initial function value
        self.percent_improvement = percent_improvement
        self.max_euclidean_value = math.sqrt(sum(element[1]**2 - element[0]**2 for element in self.domain))
        self.print_solutions = print_solutions

        n =  self.dimensions
        # Calculate the number of values (M) within each dimension
        # M_values = ((self.domain[:, 1] - self.domain[:, 0]) / self.step_size).astype(int) + 1
        # # Create a meshgrid of values for each dimension
        # solution_space = [np.linspace(self.domain[i, 0], self.domain[i, 1], M_values[i]) for i in range(n)]
        # # Convert the list of arrays to a 2D array (n x M)
        # self.solution_space = np.column_stack(solution_space)
        sol_space = []
        for i in range(self.dimensions):
            sol_space.append(np.arange(self.domain[i][0], self.domain[i][1]+ step_size[i], step_size[i]))
        
        self.solution_space = list(itertools.product(*sol_space))
        #print(self.solution_space)

    def create_neighborhood(self):
        self.neighborhoods = {}
        if self.neigh_structure==1:
            for element in self.solution_space:
                N = [el for el in self.solution_space if el != element]
                self.neighborhoods[element] = N

        else:
            discrete_solution_spaces = [set(point[dim] for point in self.solution_space) for dim in range(len(self.solution_space[0]))]
            # for dim, values in enumerate(discrete_solution_spaces):
            #     print(f"Dimension {dim}: {values}")

            #print(discrete_solution_spaces)
            #self.discrete_solution_spaces = discrete_solution_spaces
            for element in self.solution_space:
                single_space_nbd = []
                for i in range(self.dimensions):
                    single_space = list(discrete_solution_spaces[i])
                    single_space = sorted(single_space)
                    #print(single_space)
                    j = single_space.index(element[i])
                    neighbors = []
                    neighbors.append(single_space[(j - 1) % len(single_space)])
                    neighbors.append(single_space[(j + 1) % len(single_space)])
                    #print("nbd", neighbors)
                    single_space_nbd.append(neighbors)
                
                # print("element", element)
                # print(single_space_nbd)

                self.neighborhoods[element] = list(itertools.product(*single_space_nbd))

        #print(self.neighborhoods)
        return self.neighborhoods

    def get_transition_matrix(self):

        self.neighborhoods = self.create_neighborhood()
        self.D = {}
        self.R_prime = {}
        self.R = {}

        if self.neigh_structure == 1:
            for element in self.solution_space:
                self.R[element] = {}
                self.R_prime[element] = {}
                self.D[element] = 0

                for neighbor in self.neighborhoods[element]:
                    self.R_prime[element][neighbor] = self.max_euclidean_value - euclidean(element, neighbor)
                    self.D[element] += self.R_prime[element][neighbor]
        else:
            for element in self.solution_space:
                self.R[element] = {}
                self.R_prime[element] = {}
                self.D[element] = 0

                for neighbor in self.neighborhoods[element]:
                    self.R_prime[element][neighbor] = 1
                    self.D[element] += 1

        for element in self.solution_space:
            for neighbor in self.neighborhoods[element]:
                self.R[element][neighbor] = self.R_prime[element][neighbor] / self.D[element]

        #print(self.R)
        return self.R_prime, self.R
    
    def optimize(self):
        self.visits = []
        self.visits_opt = []
        self.df = pd.DataFrame()
        self.x_values = []
        self.fx_values = []
        # self.z_values = []
        # self.fz_values = []
        self.flag = -1
        if self.max_evals < 200:
            print("Error: Too less number of replications")
            return

        #self.neighborhoods = self.create_neighborhood()
        self.R_prime, self.R = self.get_transition_matrix()
        
        #for convenience convert the dictionary to an array
        position_index ={}
        index = 0

        for element in self.solution_space:
            position_index[index] = element
            index +=1
        
        
        reverse_mapping = {v: k for k, v in position_index.items()}
        #print(position_index)
        #print(len(self.solution_space))
        self.R_matrix = np.zeros((len(self.solution_space), len(self.solution_space)))
        self.R_prime_matrix = np.zeros((len(self.solution_space), len(self.solution_space)))
        for i in range(len(self.solution_space)):
            for j in range(len(self.solution_space)):
                position_i = position_index[i]
                position_j = position_index[j]
                #print(position_i, position_j)
                if position_j in self.R[position_i]:
                    self.R_matrix[i][j] = self.R[position_i][ position_j]
                    self.R_prime_matrix[i][j] =  self.R_prime[position_i][ position_j]
                else:
                    self.R_matrix[i][j] = 0
                    self.R_prime_matrix[i][j] = 0

        #print(self.R_matrix)
        if self.init_solution is None:
            if self.random_seed is None:
                random.seed(1234)
            else:
                random.seed(self.random_seed)
            random_index = random.randint(0, len(self.solution_space) - 1)
            X0 = position_index[random_index]
            self.init_solution = X0
        else:
            X0 = self.init_solution
        print("initial solution:", X0)

        self.V={}
        self.V[X0] = 1

        j = 0
        reps = []
        rep_value = 3
        self.decrease = -math.inf
        self.change = -math.inf
        starting_value = 0
        #Compute initial estimate of objective function
        for i in range(0, rep_value):
            starting_value+= self.func(X0)
        starting_value = starting_value/rep_value

        self.min_fx = starting_value
        x_next = X0
        #print(X0, starting_value, self.V[X0])
        self.x_values.append(X0)
        self.fx_values.append(starting_value)
        self.visits.append(self.V[X0])

        self.X_opt.append(X0)
        self.fx_opt.append(starting_value)
        self.visits_opt.append(self.V[X0])

        reps.append(rep_value)

        while self.max_evals>0:
            rep_value = rep_value + 1
            x_j = x_next
            N = self.neighborhoods[x_j]
            transition_probs = self.R_matrix[reverse_mapping[x_j]]
            transition_probs /= np.sum(transition_probs)
            #print(transition_probs)
            z_j = position_index[np.random.choice([i for i in range(0, len(self.solution_space))] ,p=transition_probs)]
            #print("next: ",z_j)

            fx = 0
            fz = 0
            # self.x_values.append(x_j)
            # self.z_values.append(z_j)
            simreps = min(rep_value, self.max_evals)
            for sim_iter in range(simreps):
                fx += self.func(list(x_j))
                fz += self.func(list(z_j))

            fx = fx / simreps
            fz = fz / simreps

            # self.fx_values.append(fx)
            # self.fz_values.append(fz)
            # if j == 0:
            #     self.history.append(x_j)
            #     self.fx_opt.append(fx)
            #     #self.function_values.append(fx)
            G_xz = np.exp(-(fz - fx) / self.T)
            #print(x_j, z_j)
            if np.random.rand() <= G_xz:
                x_next = z_j 
                fx_next = fz
            else :
                x_next = x_j
                fx_next = fx

            #print(x_next)
            if x_next not in self.V:
                #print("NEW SOL:", x_next)
                self.V[x_next] = 1
            else:
                #print("ALREADY EXISTS:", x_next)
                self.V[x_next] = self.V[x_next] + 1

            #print(self.V)
            #print(x_next, fx_next, self.V[x_next])
            self.fx_values.append(fx_next)
            self.x_values.append(x_next)
            self.visits.append(self.V[x_next])

            x_opt_next = x_next if self.V[x_next] / self.D[x_next] > self.V[x_next] / self.D[x_j] else x_j
            fx_opt_next = fx_next if self.V[x_next] / self.D[x_next] > self.V[x_next] / self.D[x_j] else fx
            #print(x_j, z_j, x_next, x_opt_next)
            self.max_evals = self.max_evals - simreps
            j = j+1
            reps.append(simreps)

            self.X_opt.append(x_opt_next)
            self.fx_opt.append(fx_opt_next)
            self.visits_opt.append(self.V[x_opt_next])
            self.decrease = ((starting_value - self.fx_opt[-1])/ abs(starting_value))*100
            if self.percent_improvement is not None:
                if self.decrease >= self.percent_improvement:
                    print("Stopping criterion met (% reduction in function value). Stopping optimization.")
                    self.flag = 1
                    self.x_values.append(x_opt_next)
                    self.fx_values.append(fx_opt_next)
                    self.visits.append(self.V[x_opt_next])
                    break
            self.change = max(self.change, self.decrease)

        print("Budget exhausted. Stopping optimization.")
        if self.flag == -1:
            self.x_values.append(x_opt_next)
            self.fx_values.append(fx_opt_next)
            self.visits.append(self.V[x_opt_next])
        self.min_fx = min(self.fx_opt)
        min_index = self.fx_opt.index(self.min_fx)
        self.corresponding_x_value = self.X_opt[min_index]
        print("Starting value: " , starting_value)
        if self.percent_improvement is not None:
            print("Target value: ", starting_value*(100 - self.percent_improvement)/100)
        else:
            print("No target value")

        self.visited_df = pd.DataFrame({'x': self.x_values, 'f(x)': self.fx_values, 'visits made': self.visits})
        self.df = pd.DataFrame({'x*': self.X_opt, 'f(x*)': self.fx_opt, 'reps used': reps, 'visits made': self.visits_opt})
        
    def print_function_values(self):
        if self.print_solutions:
            print("Progress of Algorithm:")
            print(self.visited_df)
            print("Optimal values:")
            print(self.df)

        if self.flag == 1:
            print("final optimal value: ", self.X_opt[-1], "final obj fn value: ",self.fx_values[-1])
            print("% reduction:", self.decrease)
        else:
            print("final optimal value: ", self.corresponding_x_value, "final obj fn value: ",self.min_fx)
            if self.init_solution == self.corresponding_x_value:
                self.change = 0.00
                print("No better solution found.")
            print("% reduction:", self.change)



def objective_function(x):
    noise = np.random.normal(scale=0.1)  # Add Gaussian noise with a standard deviation of 0.1
    return 2*x[0] + x[0]**2 + x[1]**2 + noise

#main()
dom = [[0,2], [0,2]]
step_size = [0.2, 0.2]
T= 100
k= 100



# optimizer  = SA(domain = dom, step_size= step_size, T = 100, max_evals= 200,
#                          func= objective_function, neigh_structure= 2, 
#                          random_seed= 42, percent_improvement= 40, print_solutions= True)
# optimizer.optimize()
# optimizer.print_function_values()
# print(optimizer.V)

# optimizer  = SA(domain = dom, step_size= step_size, T = 100, max_evals= 275,
#                          func= objective_function, neigh_structure= 2, 
#                          random_seed= 42)
# optimizer.optimize()
# optimizer.print_function_values()