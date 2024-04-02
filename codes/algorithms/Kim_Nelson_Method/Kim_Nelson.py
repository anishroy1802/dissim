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

class KN():

    def __init__(self, domain, step_size, func, alpha, delta, n_0= 2, max_evals= 300 ):
        self.domain = domain
        self.step_size = step_size
        self.func = func
        self.alpha = alpha
        self.delta = delta
        self.max_evals= max_evals
        self.n_0 = n_0
        self.dimensions = len(self.domain)
        sol_space = []
        for i in range(self.dimensions):
            sol_space.append(np.arange(self.domain[i][0], self.domain[i][1]+ step_size[i], step_size[i]))
        
        self.solution_space = list(itertools.product(*sol_space))
        self.confidence_lvl = 1 - self.alpha
        #self.n_systems = len(self.solution_space)
        self.eta = 0.5 * (((2*self.alpha)/(len(self.solution_space)-1))**(-2 / (self.n_0 - 1)) - 1)
        #self.r = self.n_0
        
        #print(self.solution_space)

    def initialize(self):
        a = len(self.solution_space)
        self.h = math.sqrt(2*self.eta*(self.n_0 - 1))
        self.sol_space_dict = {}
        for i in range(len(self.solution_space)):
            self.sol_space_dict[i] = self.solution_space[i]
        
        self.X_i_bar = [0]*len(self.solution_space)
        self.X_i_bar_n_0 = [0]*len(self.solution_space)
        self.S_sq = np.zeros((a, a))
        S_sq = np.zeros((a, a))
        a = len(self.solution_space)
        self.sim_vals = np.zeros((a, self.n_0)).tolist()
        #now we have a dict mapping index to an (x,y) pair
        for i in range(len(self.solution_space)):
            for simrep in range(0, self.n_0):
                self.sim_vals[i][simrep] = self.func(self.sol_space_dict[i])
                self.X_i_bar_n_0[i] += self.sim_vals[i][simrep]
            
            self.X_i_bar[i] = sum(self.sim_vals[i])
            self.X_i_bar[i] = self.X_i_bar[i]/len(self.sim_vals[i])
            self.X_i_bar_n_0[i] = self.X_i_bar_n_0[i]/self.n_0
        
        #self.X_i_bar_n_0 = self.X_i_bar[:self.n_0]
        for i in range(a):
            for j in range(a):
                S_sq[i][j] = (1/(self.n_0 - 1)) * (sum(self.sim_vals[i][0:self.n_0]) 
                                                      - sum(self.sim_vals[j][0:self.n_0])
                                                      - (self.X_i_bar_n_0[i] - self.X_i_bar_n_0[j]))**2
        
        #print(S_sq)
        return S_sq
        
    def optimize(self):
        if self.max_evals< 200:
            print("Too small budget")
            return 

        self.S_sq = self.initialize()
        r = self.n_0
        a = len(self.solution_space)
        # self.check = np.zeros((a,a))
        I_old = set()
        #Screening Procedure for Kim-Nelson Algorithm
        for i in range(0, a):
            I_old.add(i)


        #print(I_old)
        #max_evals= 0
        while len(I_old)!=1 and self.max_evals> 0:
            print("Budget left: ", self.max_evals)
            #print(self.X_i_bar)
            #print("iteration: ", k)
            print("value of r: ", r)
            print("set at beginning of iteration: ",I_old)
            #print(I_old)
            I = set()

            self.check = np.zeros((a,a))
            self.W = np.zeros((len(self.solution_space), len(self.solution_space))) 
            for i in range(len(self.solution_space)):
                for j in range(len(self.solution_space)):
                    b = (self.delta/(2*r))*(((self.h**2 * self.S_sq[i][j])/(self.delta**2)) - r)
                    self.W[i][j] = max(0, b)

            #now we have defined our W matrix
            for i in range(0,a):
                for j in range(0,a):
                    if ((i!=j) and (i in I_old) and (j in I_old)):
                        #print("a-b; a: ", self.X_i_bar[i], " b: ", self.X_i_bar[j] - self.W[i][j])
                        if self.X_i_bar[i] >= self.X_i_bar[j] - self.W[i][j]:
                            self.check[i][j] = 1

                row_sum = sum(self.check[i])
                if row_sum == len(I_old) - 1:
                    #print(i, " added")
                    I.add(i)
                # else:
                #     print(i, "not  added")
            
            #print(self.check)
            #print(I)
            # if k> 10:
            #     break
            if len(I) == 1:
                print("Got single optimal solution: ")
                return I
            else:
                I_old = I.copy()
                r += 1
                #k+= 1
                for i in range(a):
                    if i in I_old:
                        self.sim_vals[i].append(self.func(self.sol_space_dict[i]))
                        self.max_evals-= 1
                        self.X_i_bar[i]  = (self.X_i_bar[i]*(r-1) + self.sim_vals[i][-1])/r
                


        print("Final set after exhausting budget: ")   
        return I
            


# def objective_function(x):
#     noise = np.random.normal(scale=0.1)  # Add Gaussian noise with a standard deviation of 0.1
#     return -1*(2*x[0] + x[0]**2 + x[1]**2 + noise) # Minimisation problem hence -1 

# #main()
# dom = [[0,2], [0,2]]
# step_size = [0.5, 0.5]

# optimizer  = KN(domain = dom, step_size= step_size,
#                          func= objective_function,alpha=0.5, delta= 5, n_0  = 2, max_evals= 300)
# a1 = optimizer.optimize()
# #print("Solutions:")
# if a1 is not None:
#     for ele in a1:
#         print("Element: ", optimizer.sol_space_dict[ele], "; Objective fn Value: ", -1* optimizer.X_i_bar[ele])
# else:
#     print("No solution")