import math as mt
import numpy as np
import math
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from typing import Union
import tracemalloc
import pandas as pd
import dask.dataframe as dd
import time
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from itertools import product
from scipy.spatial.distance import euclidean
from typing import Union, List



def tracing_start():
    """
    Starts tracing of memory allocation using tracemalloc.
    """
    tracemalloc.stop()
    print("nTracing Status : ", tracemalloc.is_tracing())
    tracemalloc.start()
    print("Tracing Status : ", tracemalloc.is_tracing())
def tracing_mem():
    """
    Prints the peak memory usage.
    """
    first_size, first_peak = tracemalloc.get_traced_memory()
    peak = first_peak/(1024*1024)
    print("Peak Size in MB - ", peak)


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

"""
Kin Nelson Algorithm
"""

class KN():

    def __init__(self, domain, step_size, func, alpha, delta, n_0= 2, max_evals= 300, print_solutions = True):
        self.domain = domain
        self.step_size = step_size
        self.func = func
        self.alpha = alpha
        self.delta = delta
        self.max_evals= max_evals
        self.n_0 = n_0
        self.dimensions = len(self.domain)
        sol_space = []
        self.print_solutions = print_solutions
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
        fx_values = []
        x_values = []
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
            # print("Budget left: ", self.max_evals)
            # #print(self.X_i_bar)
            # #print("iteration: ", k)
            # print("value of r: ", r)
            # print("set at beginning of iteration: ",I_old)
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
                for ele in I:
                    x_values.append(self.sol_space_dict[ele])
                    fx_values.append(self.X_i_bar[ele])
                self.df = pd.DataFrame({'x*': x_values, 'f(x*)': fx_values})
                if self.print_solutions:
                    print(self.df)
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

        if I is not None:
            for ele in I:
                x_values.append(self.sol_space_dict[ele])
                fx_values.append(self.X_i_bar[ele])
            self.df = pd.DataFrame({'x*': x_values, 'f(x*)': fx_values})

            if self.print_solutions:
                print(self.df)
        return I

"""
Simulated Anealing Algorithm
"""
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
        # if self.flag == -1:
        #     self.x_values.append(x_opt_next)
        #     self.fx_values.append(fx_opt_next)
        #     self.visits.append(self.V[x_opt_next])
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

"""
Adaptive Hyperbox Algorithm
"""

class AHA():
  def __init__(self,func,domain, step_size, max_evals, init_solution, m, random_seed  = None, percent_improvement = None, print_solutions = True):
    """Constructor method that initializes the instance variables of the class AHA

    Args:
      func (function): A function that takes in a list of integers and returns a float value representing the objective function.
      domain (list of lists): A list of intervals (list of two integers) that defines the search space for the optimization problem.
    """
    self.func = func
    self.percent_improvement = percent_improvement
    self.domain = domain
    self.random_seed = random_seed
    self.step_size = step_size
    self.max_evals = max_evals
    self.init_solution = init_solution
    self.m = m
    # self.initial_choice = initial_choice
    self.x_star = []
    self.print_solutions = print_solutions
    #self.initial_choice = self.sample(domain)
    sol_space = []
    self.dimensions  = len(self.step_size)
    for i in range(self.dimensions):
        sol_space.append(np.arange(self.domain[i][0], self.domain[i][1]+ step_size[i], step_size[i]))
    
    self.solution_space = list(itertools.product(*sol_space))
    
  def sample(self, domain):
    """A helper method that samples a point from the given domain. 
      It returns a list of integers representing a point in the search space.

    Args:
        domain (list of lists): A list of intervals (list of two integers) to sample from.

    Returns:
        list: A list of integers representing a point in the search space.
    """
    x = []
    for i,d in enumerate(domain): #discrete domain from a to b
      a,b = d[0],d[1]
      stepsize = self.step_size[i]
      num_steps = (b - a) // stepsize + 1
      random_index = np.random.randint(num_steps)
      random_number = a + random_index * stepsize

      x.append(random_number)
    
    return x

  def ak(self,iters : int):
    """ A method that computes the value of the parameter 'a' for the given iteration.

    Args:
        iters (int): The current iteration number.

    Returns:
        int: The value of the parameter 'a'.
    """
    if iters <= 2.0: a = 3
    else: a = min(3, mt.ceil(3*(mt.log(iters))**1.01))
    return a

  #finding l_k u_k
  def hk_mpa(self,xbest_k,f,l_k,u_k):
    """Computes the values of 'l_k' and 'u_k' for the given iteration.

    Args:
        xbest_k (list): The current best solution found.
        f (list of lists): A list of unique solutions found so far.
        l_k (list): A list of lower bounds for each dimension in the search space.
        u_k (list): A list of upper bounds for each dimension in the search space.

    Returns:
        tuple: A tuple containing the updated values of 'l_k' and 'u_k'.
    """
    for i in range(len(xbest_k)):
        #f is the set of unique solutions
        for j in range(len(f)):
            if xbest_k != f[j] and l_k[i] <= f[j][i]:
                if xbest_k[i] > f[j][i]:
                    l_k[i] = f[j][i]
            if xbest_k != f[j] and u_k[i] >= f[j][i]:
                if xbest_k[i] < f[j][i]:
                    u_k[i] = f[j][i]
    return l_k, u_k

  def ck(self,l_k, u_k):
    """Computes the set of candidate intervals 'c_k'.

    Args:
        l_k (list): A list of lower bounds for each dimension in the search space.
        u_k (list): A list of upper bounds for each dimension in the search space.

    Returns:
        list of lists: A list of intervals (list of two integers) representing the set of candidate intervals 'c_k'.
    """
    mpa_k = []
    for i in range(len(l_k)):
        mpa_k.append([l_k[i],u_k[i]])
    return mpa_k      



  def optimize(self):
    """ Runs the AHA algorithm for local optimization.

    Returns:
        list of lists: A list of solutions found by the algorithm at each iteration.
    """
    #initialisation
    # print(self.init_solution)
    m = self.m
    if self.max_evals < 200:
       print("Error: too less simulation budget")
    tracing_start()
    start = time.time()
    if self.init_solution is None:
       if self.random_seed is None:
            random.seed(1234)
       else:
            random.seed(self.random_seed)
       self.init_solution = random.choice(self.solution_space)
       
    self.fxvals = []
    self.fx_star = []
    self.all_x = []
    self.all_fx = []
    self.x_star.append(self.init_solution)
    #self.initval = self.func(self.init_solution)
    
    #print(self.initval)
    epsilon = []
    epsilon.append([self.init_solution])
    G = 0
    self.initval = 0
    for i in range(5):
      G += self.func(self.init_solution)
      self.initval += self.func(self.init_solution)
    G_bar_best = G/5
    self.initval = self.initval/5
    

    self.max_evals = self.max_evals - 5
    l_k = []
    u_k = []
    for i in range(len(self.domain)):
        l_k.append(self.domain[i][0])
        u_k.append(self.domain[i][1])

    # c = [domain]
    # phi = domain

    all_sol = [self.init_solution]
    self.fx_star.append(self.initval)
    all_fx = [self.initval]
    all_x = [self.init_solution]
    uniq_sol_k =[]
    while self.max_evals > 0:
      k = 0
      all_sol_k = []
      
      xk =[]
      hk=[]

      l_k,u_k = self.hk_mpa(self.x_star[k-1],uniq_sol_k,l_k,u_k)
      ck_new = self.ck(l_k,u_k)  


      for i in range(m):
        xkm = self.sample(ck_new)
        # xk.append(xkm)
        all_sol_k.append(xkm)

      uniq_sol_k = list(set(tuple(x) for x in all_sol_k))
      # print(uniq_sol_k)
      all_sol.append(all_sol_k)
      # print(uniq_sol_k)
      epsilonk = uniq_sol_k + [tuple(self.x_star[k-1])]
      # print(epsilonk)
      x_star_k = self.x_star[k-1]
      #print(epsilonk)


      for i in epsilonk:
        #print(list(i))
        all_x.append(list(i))
        k+= 1
        numsimreps = min(5, self.max_evals)
        self.max_evals = self.max_evals - numsimreps
        g_val = 0
        for j in range(numsimreps):
          g_val += self.func(i)

        g_val_bar = g_val/numsimreps
        all_fx.append(g_val_bar)

        if(G_bar_best>g_val_bar):
          G_bar_best = g_val_bar
          x_star_k = list(i)
          self.x_star.append(x_star_k)
          self.fx_star.append(G_bar_best)
        
        # all_fx.append(G_bar_best)
        # all_x.append(x_star_k)
        # print("all_fx")
        # print(all_x, all_fx)
        self.decrease = 100*((self.initval - all_fx[-1] ))/ abs(self.initval)
        if ((self.percent_improvement is not None) and (self.decrease >= self.percent_improvement)):
          print("Stopping criterion met (% reduction in function value). Stopping optimization.")
          # self.x_star.append(x_star_k)
          # self.fx_star.append(G_bar_best)
          self.all_x = all_x
          self.all_fx = all_fx
          return self.x_star

        if self.max_evals == 0:
          print("Budget exhausted. Stopping optimization.")
          self.all_x = all_x
          self.all_fx = all_fx
          # self.x_star.append(x_star_k)
          # self.fx_star.append(G_bar_best)
          return self.x_star
          

      #self.decrease = 100*((self.initval - self.fx_star[-1] ))/ abs(self.initval)
      #epsilon.append(epsilonk)
      #k += 1
      #self.max_evals = self.max_evals - numsimreps
      # if ((self.percent_improvement is not None) and (self.decrease >= self.percent_improvement)):
      #   print("Stopping criterion met (% reduction in function value). Stopping optimization.")
      #   break

    #self.fxvals = all_fx

    end = time.time()
    print("time elapsed {} milli seconds".format((end-start)*1000))
    tracing_mem()
    
    return self.x_star
  
  def print_function_values(self):

    self.df = pd.DataFrame({'x': self.all_x, 'f(x)': self.all_fx})
    if self.print_solutions:
      print (self.df)
    print("iters:",len(self.all_x)-1)
    print("initial x: ", self.init_solution, "initial fx estimated: ", self.initval)
    print("optimal x* values: ", self.x_star)
    print("optimal f(x*) values: ",self.fx_star)
    print("% decrease:", self.decrease )
    #func_values = [self.fxvals(x) for x in self.x_star]  # Evaluate the function for each solution
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(len(self.x_star)), func_values, marker='o', linestyle='-')
    # plt.xlabel('Iteration')
    # plt.ylabel('Function Value')
    # plt.title('Function Value vs. Iteration')
    # plt.grid(True)
    # plt.show()


  # def AHAalgoglobal(self,iter,max_k,m):
  #   """ Runs the AHA algorithm for global optimization.

  #   Args:
  #       iter (int): The number of iterations to run.
  #   max_k (int): The maximum number of iterations to run for each iteration of the global search.
  #   m (int): The number of random samples to generate at each iteration.

  #   Returns:
  #       tuple: A tuple containing the best solution found and its corresponding objective value.
  #   """
  #   all_sols = []
  #   best_sol = None
  #   best_val = None

  #   # for i in range(iter):
  #   #   new_dom = []
  #   #   for dom in self.domain:
  #   #     if(dom[0]!=dom[1]):
  #   #       val1 = np.random.randint(dom[0],dom[1]+1)
  #   #       val2 = np.random.randint(dom[0],dom[1]+1)
  #   #       while(val1==val2):
  #   #         val2 = np.random.randint(dom[0],dom[1]+1)
  #   #       if(val1<val2):
  #   #         new_dom.append([val1,val2])
  #   #       else:
  #   #         new_dom.append([val2,val1])
  #   #     else:
  #   #       new_dom.append(dom)

  #   for i in range(iter):
  #     new_dom = []
  #     for dom in self.domain:
  #       D = (dom[1] - dom[0])
  #       if(D>=2 and D<=20):
  #         K=3
  #       elif(D>20 and D<=50):
  #         K=5
  #       elif(D>50 and D<=100):
  #         K = 10
  #       elif(D>100):
  #         K = 50



  #       if(dom[1]-dom[0]>2):
  #         step = (dom[1]-dom[0])//K
  #         val1 = np.random.randint(dom[0],dom[1]+1)
  #         if(val1+step+1<dom[1]):
  #           val2 = np.random.randint(val1+1,val1+step+1)
  #         else:
  #           val2 = np.random.randint(val1-step-1,val1)
  #         if(val1<val2):
  #           new_dom.append([val1,val2])
  #         else:
  #           new_dom.append([val2,val1])


  #       elif(dom[0]!=dom[1]):
  #         val1 = np.random.randint(dom[0],dom[1]+1)
  #         val2 = np.random.randint(dom[0],dom[1]+1)
  #         while(val1==val2):
  #           val2 = np.random.randint(dom[0],dom[1]+1)
  #         if(val1<val2):
  #           new_dom.append([val1,val2])
  #         else:
  #           new_dom.append([val2,val1])
  #       else:
  #         new_dom.append(dom)

  #     # print(new_dom)
  #     self.init_solution = self.sample(new_dom)
  #     solx = self.optimize(max_k,m,new_dom,self.init_solution)

  #     valx = self.func(solx[-1])
  #     if(best_sol == None or valx<best_val):
  #       best_sol = solx[-1]
  #       best_val = valx
  #     all_sols.append(solx[-1])

  #   return all_sols,best_sol,best_val
  
  """
  Stochastic Ruler Method
  """

class stochastic_ruler:
    """
    The class definition for the implementation of the Stochastic Ruler Random Search Method;
    Alrefaei, Mahmoud H., and Sigrun Andradottir.
    "Discrete stochastic optimization via a modification of the stochastic ruler method."
    Proceedings Winter Simulation Conference. IEEE, 1996.
    """

    def __init__( self, space: dict, max_evals: int = 300, prob_type="opt_sol", func=None, 
        percent_improvement: int = None, init_solution: dict = None, lower_bound: int = None, 
        upper_bound: int = None, neigh_structure : int = 2, print_solutions: bool = False):
        """The constructor for declaring the instance variables in the Stochastic Ruler Random Search Method

        Args:
            space (dict): allowed set of values for the set of hyperparameters in the form of a dictionary
                        hyperparamater name -> key
                        the list of allowed values for the respective hyperparameter -> value
            max_evals (int, optional): maximum number of evaluations for the performance measure; Defaults to 100.
        """
        self.space = space  # domain
        self.prob_type = ( prob_type ) # hyperparam opt (hyp_opt), optimal solution (opt_sol) 
        self.data = None
        self.max_evals = max_evals
        self.initial_choice_HP = None
        self.Neigh_dict = self.help_neigh_struct()
        self.func = func
        self.percent_improvement = percent_improvement
        self.init_solution = init_solution
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.neigh_structure = neigh_structure
        self.print_solutions = print_solutions

    def help_neigh_struct(self) -> dict:
        Dict = {}
        for hyper_param in self.space:
            for i, l in enumerate(self.space[hyper_param]):
                key = hyper_param + "_" + str(l)
                Dict[key] = i
        return Dict

        """
    The helper method for creating a dictionary containing the position of the respective hyperparameter value in the enumered dictionary of space

    Returns:
        dict: hyperpamatername concatenated with its value ->key
              zero-based index position of the hyperparameter value in self.space (the hype) -> value
    """


    #N2
    def random_pick_from_neighbourhood_structure(self, initial_choice: dict) -> dict:
        set_hp = {}
        for hp in initial_choice:
            key = str(hp) + "_" + str(initial_choice[hp])
            hp_index = self.Neigh_dict[key]
            idx = random.choice([-1, 1])
            length = len(self.space[hp])
            set_hp[hp] = self.space[hp][(hp_index + idx + length) % length]
        return set_hp



    #N1
    def next_solution_based_on_distance(self, initial_solution: dict) -> dict:
        all_combinations = []
        distances = []

        # Generate all combinations in the neighborhood
        for hp in initial_solution:
            local_neighborhood = [val for val in self.space[hp] if val != initial_solution[hp]]
            all_combinations.append(local_neighborhood)

        all_combinations = list(product(*all_combinations))

        for combination in all_combinations:
            potential_solution = dict(zip(initial_solution.keys(), combination))
            # Calculate the Euclidean distance between the current solution and the potential solution
            current_values = list(initial_solution.values())
            potential_values = list(potential_solution.values())
            dist = euclidean(current_values, potential_values)
            distances.append(dist)

        # Normalize distances to create a probability distribution
        probabilities = [dist/sum(distances) for dist in distances]

        # Choose a new solution based on the calculated probabilities
        chosen_index = np.random.choice(len(all_combinations), p=probabilities)
        chosen_combination = all_combinations[chosen_index]
        return dict(zip(initial_solution.keys(), chosen_combination))


    # def no_of_solutions_visited(self, max_evals):

    #     """The method for calculating the maximum solutions that can be visited from the imput budget
    #     Args:
    #         max_evals (int): the budget in terms of simulations

    #     Returns:
    #         int: the maximum number of solutions that can be visited as a part of the algorithm
    #     """

    #     sols = -1
    #     budget_exhausted=0
    #     while budget_exhausted <= max_evals:
    #         budget_exhausted+=int(math.log(sols + 10, math.e) / math.log(5, math.e))
    #         sols+=1

    #     return sols


    def det_a_b(self, domain, max_eval, X=None, y=None):
        """Computes the minimum and maximum values of the function represented by self,
        using Stochastic Ruler with random samples from the given domain. This gives us (a,b) of the stochastic ruler

        Args:
            domain (dict): A dictionary that maps the names of the variables of the function represented by self to their domains,
            which should be represented as lists or arrays of values.
            max_eval (int): The maximum number of evaluations of the function to perform. The total number of evaluations will be
            approximately max_eval, but may be slightly lower due to the fact that each iteration involves
            len(domain) evaluations.
            X (array-like or None, optional): An array of input values to pass to the function represented by self. Defaults to None.
            y (array-like or None, optional): An array of target values to pass to the function represented by self. Defaults to None.

        Returns:
            tuple: This gives us (a,b) of the stochastic ruler
        """
        if self.lower_bound is not None and self.upper_bound is not None:
            minm = self.lower_bound
            maxm = self.upper_bound

        else:
            max_iter = 20
        
            Reps_for_each_sol = 5
            maxm = -math.inf
            minm = math.inf

            for i in range(max_iter):
                # print("i = ", i)
                for j in range(Reps_for_each_sol):
                    # print("j = ", j)
                    # Randomly sample from the domain for each variable
                    neigh = {var: np.random.choice(values) for var, values in domain.items()}
                    temp = self.run(neigh, neigh, X, y)
                    minm = min(minm, temp)
                    maxm = max(maxm, temp)
                    #print(minm, maxm)

        return (minm, maxm)



    def Mf(self, k: int) -> int:
        """The method for represtenting the maximum number of failures allowed while iterating for the kth step in the Stochastic Ruler Method
            In this case, we let Mk = floor(log_5 (k + 10)) for all k; this choice of the sequence {Mk} satisfies the guidelines specified by Yan and Mukai (1992)
        Args:
            k (int): the iteration number in the Stochastic Ruler Method

        Returns:
            int: the maximum number for the number of iterations
        """

        return int(3+math.log(k + 10, math.e) / math.log(5, math.e)) 


    


    def SR_Algo(self, X: np.ndarray = None, y: np.ndarray = None) -> Union[float, dict, float, float, List[float]]:
        """The method that uses the Stochastic Ruler Method (Yan and Mukai 1992)
           Step 0: Select a starting point Xo E S and let k = 0.
           Step 1: Given xk = x, choose a candidate zk from N(x) randomly
           Step 2: Given zk = z, draw a sample h(z) from H(z).
           Then draw a sample u from U(a, b). If h(z) > u, then let X(k+1) = Xk and go to Step 3.
           Otherwise dmw another sample h(z) from H(z) and draw another sample u from U(a, b).
           If h(z) > u, then let X(k+1) = Xk and go to Step3.
           Otherwise continue to draw and compare.
           If all Mk tests, h(z) > u, fail, then accept the candidate zk and Set xk+1 = zk = Z.

        Args:
            X (np.ndarray): feature matrix
            y (np.ndarray): label

        Returns:
            Union[float,dict]: the minimum value of 1-accuracy and the corresponding optimal hyperparameter set
        """
        # X_train,X_test,y_train,y_test = self.data_preprocessing(X,y)

        self.minh_of_z_tracker = []

        if self.percent_improvement is not None:

            initial_choice_HP = {}

            if self.init_solution is None:
                for i in self.space:
                    # initial_choice_HP[i] = self.space[i][0]
                    initial_choice_HP[i] = random.choice(self.space[i])
                print("initial solution = ",initial_choice_HP)
            else:
                initial_choice_HP = self.init_solution

            # printing initial value for checking
            # init_value = self.run(initial_choice_HP, initial_choice_HP, X, y)
            init_value = self.func(initial_choice_HP)
            print("Initial value = ", init_value)

            self.target_value = init_value * (1 - self.percent_improvement * 0.01) if init_value >= 0 else init_value * (1 + self.percent_improvement * 0.01)
            print("Target Value = ", self.target_value)
            print("------")

            # step 0: Select a starting point x0 in S and let k = 0
            k = 0
            x_k = initial_choice_HP
            opt_x = x_k
            a, b = self.det_a_b(self.space, self.max_evals // 10, X, y)
            # print("a,b:")
            # print(a,b)
            minh_of_z = b
            # step 0 ends here
            # print("total evals: ", self.no_of_solutions_visited(self.max_evals))
            while k < self.max_evals:
                
                # step 1:  Given xk = x, choose a candidate zk from N(x)
                if self.neigh_structure == 1:
                    zk = self.next_solution_based_on_distance(x_k)                  #N1
                
                elif self.neigh_structure == 2:
                    zk = self.random_pick_from_neighbourhood_structure(x_k)         #N2


                # step 1 ends here
                # step 2: Given zk = z, draw a sample h(z) from H(z)
                iter = self.Mf(k)
                
                for i in range(iter):
                    h_of_z = self.run(zk, zk, X, y)
                    
                    # print("value at iter: ", h_of_z)

                    if self.print_solutions:
                        print("k: " , k, "x_k: ", x_k, "f(x_k): ", h_of_z )  
                    k+=1 

                    if h_of_z <= self.target_value :
                        # print("h of z at stop: ", h_of_z)

                        print("Stopping criterion of ", self.percent_improvement,"% reduction in function value. Stopping optimization.")
                        # return h_of_z, opt_x, a, b, minh_of_z_tracker
                        self.minh_of_z_tracker.append(h_of_z)
                        # print(self.minh_of_z_tracker)
                        return h_of_z, opt_x, a, b

                    u = np.random.uniform(a, b)  # Then draw a sample u from U(a, b)

                    # if h_of_z > u:  # If h(z) > u, then let xk+1 = xk and go to step 3.
                    #     # k += 1
                    #     if h_of_z < minh_of_z:          # not a part of SR, comment for now. 
                    #         minh_of_z = h_of_z
                    #         opt_x = x_k
                    #     k+=1 
                    #     break
                    # Otherwise draw another sample h(z) from H(z) and draw another sample u from U(a, b), part of the loop where iter = self.Mf(k) tells the maximum number of failures allowed
                    if h_of_z <= u:  # If all Mk tests have failed
                        x_k = zk
                        # k += 1
                        if h_of_z < minh_of_z:
                            minh_of_z = h_of_z
                            self.minh_of_z_tracker.append(h_of_z)
                            opt_x = zk
                    
                # step 2 ends here
                # step 3: k = k+1

            # return minh_of_z, opt_x, a, b, minh_of_z_tracker
            return minh_of_z, opt_x, a, b 

        else:
            print("No percent Reduction criteria set:")
            initial_choice_HP = {}

            if self.init_solution is None:
                for i in self.space:
                    # initial_choice_HP[i] = self.space[i][0]
                    initial_choice_HP[i] = random.choice(self.space[i])
                print("initial solution = ",initial_choice_HP)
            else:
                initial_choice_HP = self.init_solution

            # printing initial value for checking
            init_value = self.func(initial_choice_HP)


            # step 0: Select a starting point x0 in S and let k = 0
            k = 0
            x_k = initial_choice_HP
            opt_x = x_k
            a, b = self.det_a_b(self.space, self.max_evals // 10, X, y)

            minh_of_z = b
            index = 0
            # step 0 ends here
            while k < self.max_evals:

                f_avg = 0

                # step 1:  Given xk = x, choose a candidate zk from N(x)

                if self.neigh_structure == 1:
                    zk = self.next_solution_based_on_distance(x_k)                  #N1
                
                elif self.neigh_structure == 2:
                    zk = self.random_pick_from_neighbourhood_structure(x_k)         #N2
                # step 1 ends here
                    
                # step 2: Given zk = z, draw a sample h(z) from H(z)
                iter = self.Mf(k)
                for i in range(iter):
                    h_of_z = self.run(zk, zk, X, y)
                    f_avg+=h_of_z
                    
                    # print(h_of_z)

                    # if self.print_solutions:
                    #     print("k: " , k, "x_k: ", x_k, "f(x_k): ", h_of_z )
                    
                    k += 1

                    u = np.random.uniform(a, b)  # Then draw a sample u from U(a, b)

                    if h_of_z < u:  # If all Mk tests have failed
                        x_k = zk
                        if h_of_z < minh_of_z:
                            minh_of_z = h_of_z
                            self.minh_of_z_tracker.append(minh_of_z)
                            opt_x = zk
                    
                index +=1
                print("iters = ", iter, "sol no: ", index, "x_k = ", x_k, "z_k = ", zk, "avg value at zk = ", f_avg/iter)
                       
                # step 2 ends here
                # step 3: k = k+1


            return minh_of_z, opt_x, a, b

    def optsol(self):
        """this gives the optimal solution for the problem using SR_Algo() method

        Returns:
            Union [float, dict]: the optimal solution represented as a dictionary and the corresponding value in float/int
        """
        # tracing_start()
        start = time.time()
        result = self.SR_Algo()

        end = time.time()
        print("time elapsed {} milli seconds".format((end - start) * 1000))
        # tracing_mem()

        return result

    def run( self, opt_x, neigh: dict, X: np.ndarray = None, y: np.ndarray = None) -> float:
        """The (helper) method that instantiates the model function called from sklearn and returns the additive inverse of accuracy to be minimized

        Args:
            neigh (dict): helper dictionary with positions as values and the concatenated string of hyperparameter names and their values as their keys
            X (np.ndarray): Feature Matrix in the form of numpy arrays
            y (np.ndarray): label in the form of numpy arrays

        Returns:
            float: the additive inverse of accuracy to be minimized
        """

        if self.prob_type == "opt_sol":
            funcval = self.func(opt_x)
            return funcval
        # print("acc" + str(acc))
