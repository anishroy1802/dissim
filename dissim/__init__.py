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


### Simulated Annealing algorithm


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

"""Memory Tracing:
 The 'tracing_start' function is defined to manage memory tracing. It begins by stopping any ongoing memory tracing using 'tracemalloc.stop()' and checking if tracing is currently active with 'tracemalloc.is_tracing()'. Afterward, it starts memory tracing with 'tracemalloc.start()' and checks the tracing status again. This function is invaluable for profiling memory usage, enabling developers to gain insights into memory allocation and deallocation patterns during code execution. It can help identify potential memory-related issues and optimize memory usage in the application."""

def tracing_start():
    tracemalloc.stop()
    print("Tracing Status: ", tracemalloc.is_tracing())
    tracemalloc.start()
    print("Tracing Status: ", tracemalloc.is_tracing())

class SA():

    def __init__(self, domain, step_size, T, custom_H_function, nbd_structure, k= 300, percent_reduction=None):

        self.domain = domain #n*2 matrix
        self.dimensions = len(self.domain)
        print("Number of dimensions:", self.dimensions)
        self.step_size = step_size #n*1 matrix
        self.T = T
        self.k = k
        self.custom_H_function = custom_H_function
        self.L = [i + 100 for i in range(0, k)]
        self.X_opt = []
        self.nbd_structure = nbd_structure
        self.function_values = []
        self.initial_function_value = None  # Store the initial function value
        self.percent_reduction = percent_reduction

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
        if self.nbd_structure=="N1":
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

        if self.nbd_structure == "N1":
            for element in self.solution_space:
                self.R[element] = {}
                self.R_prime[element] = {}
                self.D[element] = 0

                for neighbor in self.neighborhoods[element]:
                    self.R_prime[element][neighbor] = euclidean(element, neighbor)
                    self.D[element] += euclidean(element, neighbor)
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
        # self.neighborhoods = self.create_neighborhood()
        # self.D = {}
        # self.R_prime = {}
        # self.R = {}
        # """
        # initialize R matrix and R_prime matrix as a dictionary to store values of R(x, x') 
        # for x' being a neighbour of x and vice versa
        # """
        # for element in self.solution_space:
        #     self.R[element] = {}
        #     self.R_prime[element] = {}
        
        # for element in self.solution_space:
        #     if self.neighborhoods[element] is not None:
        #         for neighbor in self.neighborhoods[element]:
        #             if self.nbd_structure=="N1":
        #                 self.R_prime[element][neighbor] = euclidean(element, neighbor)
        #                 self.D[element] += euclidean(element, neighbor)
        #             else:
        #                 self.R_prime[element][neighbor] = 1
        #                 self.D[element] += 1
        
        # for element in self.solution_space:
        #     self.R[element][neighbor] = self.R_prime[element][neighbor]/self.D[element]

        # return self.R_prime, self.R

    def optimize(self):
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
        random_index = random.randint(0, len(self.solution_space) - 1)
        X0 = position_index[random_index]
        print("initial solution:", X0)

        self.V={}
        self.V[X0] = 1
        self.X_opt.append(X0)

        for j in range(self.k):
            x_j = self.X_opt[j]
            N = self.neighborhoods[x_j]
            transition_probs = self.R_matrix[reverse_mapping[x_j]]
            transition_probs /= np.sum(transition_probs)
            #print(transition_probs)
            z_j = position_index[np.random.choice([i for i in range(0, len(self.solution_space))] ,p=transition_probs)]
            #print("next: ",z_j)

            fx = 0
            fz = 0

            for sim_iter in range(self.L[j]):
                fx += self.custom_H_function(list(x_j))
                fz += self.custom_H_function(list(z_j))

            fx = fx / self.L[j]
            fz = fz / self.L[j]
            G_xz = np.exp(-(fz - fx) / self.T)
            x_next = z_j if np.random.rand() <= G_xz else x_j
            if x_next not in self.V:
                self.V[x_next] = 1
            else:
                self.V[x_next] += 1

            D_x_j = self.D[x_j]
            D_x_next = self.D[x_next]
            x_opt_next = x_next if self.V[x_next] / D_x_next > self.V[x_next] / self.D[x_j] else x_j
            self.X_opt.append(x_opt_next)

            # Calculate and store the function value for this iteration
            current_function_value = self.custom_H_function(list(x_opt_next))
            self.function_values.append(current_function_value)

            #if self.percent_reduction is not None:
            if self.initial_function_value is None:
                self.initial_function_value = current_function_value
            else:
                if self.percent_reduction is not None:
                    if ((self.initial_function_value - current_function_value) >= abs(self.initial_function_value) * (self.percent_reduction/100)):
                        print("Stopping criterion of " ,self.percent_reduction,   "% reduction in function value). Stopping optimization.")
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
        #print("V matrix:", self.V)
        print("iters:", len(self.function_values))
        #print("VALUES:")
        # print("init:", self.function_values[0])
        # print("final:", self.function_values[-1])
        print("final optimal value: ", self.X_opt[-1])
        print("% reduction:",  100*(self.function_values[0] -  self.function_values[-1])/abs(self.function_values[0]))
        #print(self.function_values)

### Adaptive Hyperbox Algorithm

def tracing_start():
    tracemalloc.stop()
    print("nTracing Status : ", tracemalloc.is_tracing())
    tracemalloc.start()
    print("Tracing Status : ", tracemalloc.is_tracing())
def tracing_mem():
    first_size, first_peak = tracemalloc.get_traced_memory()
    peak = first_peak/(1024*1024)
    print("Peak Size in MB - ", peak)

class AHA():
  def __init__(self, func,global_domain):
    """Constructor method that initializes the instance variables of the class AHA

    Args:
      func (function): A function that takes in a list of integers and returns a float value representing the objective function.
      global_domain (list of lists): A list of intervals (list of two integers) that defines the search space for the optimization problem.
    """
    self.func = func
    self.global_domain = global_domain
    # self.initial_choice = initial_choice
    self.x_star = []
    #self.initial_choice = self.sample(global_domain)
    
  def sample(self,domain):
    """A helper method that samples a point from the given domain. 
      It returns a list of integers representing a point in the search space.

    Args:
        domain (list of lists): A list of intervals (list of two integers) to sample from.

    Returns:
        list: A list of integers representing a point in the search space.
    """
    x = []
    for d in domain: #discrete domain from a to b
      a,b = d[0],d[1]
      x.append(np.random.randint(a,b+1))
    
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



  def AHAalgolocal(self,max_k,m,loc_domain,x0):
    """ Runs the AHA algorithm for local optimization.

    Args:
        max_k (int): The maximum number of iterations to run.
        m (int): The number of random samples to generate at each iteration.
        loc_domain (list of lists): A list of intervals (list of two integers) that defines the local search space.
        x0 (list): A list of integers representing the initial solution.

    Returns:
        list of lists: A list of solutions found by the algorithm at each iteration.
    """
    #initialisation
    # print(x0)
    self.x_star.append(x0)
    epsilon = []
    epsilon.append([x0])
    G = 0
    for i in range(self.ak(0)):
      G += self.func(x0)
    G_bar_best = G/self.ak(0)

    l_k = []
    u_k = []
    for i in range(len(loc_domain)):
        l_k.append(loc_domain[i][0])
        u_k.append(loc_domain[i][1])

    # c = [domain]
    # phi = domain

    all_sol = [x0]
    uniq_sol_k =[]
    for k in range(1,max_k+1):
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
      for i in epsilonk:
        numsimobs = self.ak(k)
        g_val = 0
        for j in range(numsimobs):
          g_val += self.func(i)
        g_val_bar = g_val/numsimobs
        if(G_bar_best>g_val_bar):
          G_bar_best = g_val_bar
          x_star_k = list(i)
      self.x_star.append(x_star_k)
      epsilon.append(epsilonk)


    return self.x_star
  def AHAalgoglobal(self,iter,max_k,m):
    """ Runs the AHA algorithm for global optimization.

    Args:
        iter (int): The number of iterations to run.
    max_k (int): The maximum number of iterations to run for each iteration of the global search.
    m (int): The number of random samples to generate at each iteration.

    Returns:
        tuple: A tuple containing the best solution found and its corresponding objective value.
    """
    all_sols = []
    best_sol = None
    best_val = None

    # for i in range(iter):
    #   new_dom = []
    #   for dom in self.global_domain:
    #     if(dom[0]!=dom[1]):
    #       val1 = np.random.randint(dom[0],dom[1]+1)
    #       val2 = np.random.randint(dom[0],dom[1]+1)
    #       while(val1==val2):
    #         val2 = np.random.randint(dom[0],dom[1]+1)
    #       if(val1<val2):
    #         new_dom.append([val1,val2])
    #       else:
    #         new_dom.append([val2,val1])
    #     else:
    #       new_dom.append(dom)

    for i in range(iter):
      new_dom = []
      for dom in self.global_domain:
        D = (dom[1] - dom[0])
        if(D>=2 and D<=20):
          K=3
        elif(D>20 and D<=50):
          K=5
        elif(D>50 and D<=100):
          K = 10
        elif(D>100):
          K = 50



        if(dom[1]-dom[0]>2):
          step = (dom[1]-dom[0])//K
          val1 = np.random.randint(dom[0],dom[1]+1)
          if(val1+step+1<dom[1]):
            val2 = np.random.randint(val1+1,val1+step+1)
          else:
            val2 = np.random.randint(val1-step-1,val1)
          if(val1<val2):
            new_dom.append([val1,val2])
          else:
            new_dom.append([val2,val1])


        elif(dom[0]!=dom[1]):
          val1 = np.random.randint(dom[0],dom[1]+1)
          val2 = np.random.randint(dom[0],dom[1]+1)
          while(val1==val2):
            val2 = np.random.randint(dom[0],dom[1]+1)
          if(val1<val2):
            new_dom.append([val1,val2])
          else:
            new_dom.append([val2,val1])
        else:
          new_dom.append(dom)

      # print(new_dom)
      x0 = self.sample(new_dom)
      solx = self.AHAalgolocal(max_k,m,new_dom,x0)

      valx = self.func(solx[-1])
      if(best_sol == None or valx<best_val):
        best_sol = solx[-1]
        best_val = valx
      all_sols.append(solx[-1])

    return all_sols,best_sol,best_val
  


#### Stochastic Ruler Algorithm

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

import numpy as np
import math
from typing import Union
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
from itertools import product
from scipy.spatial.distance import euclidean
from typing import Union, List


class stochastic_ruler:
    """
    The class definition for the implementation of the Stochastic Ruler Random Search Method;
    Alrefaei, Mahmoud H., and Sigrun Andradottir.
    "Discrete stochastic optimization via a modification of the stochastic ruler method."
    Proceedings Winter Simulation Conference. IEEE, 1996.
    """

    def __init__( self, space: dict, maxevals: int = 300, prob_type="opt_sol", func=None, 
        percentReduction: int = None, init_solution: dict = None, lower_bound: int = None, 
        upper_bound: int = None, neigh_structure : int = 1):
        """The constructor for declaring the instance variables in the Stochastic Ruler Random Search Method

        Args:
            space (dict): allowed set of values for the set of hyperparameters in the form of a dictionary
                        hyperparamater name -> key
                        the list of allowed values for the respective hyperparameter -> value
            maxevals (int, optional): maximum number of evaluations for the performance measure; Defaults to 100.
        """
        self.space = space  # domain
        self.prob_type = ( prob_type ) # hyperparam opt (hyp_opt), optimal solution (opt_sol) 
        self.data = None
        self.maxevals = maxevals
        self.initial_choice_HP = None
        self.Neigh_dict = self.help_neigh_struct()
        self.func = func
        self.percentReduction = percentReduction
        self.init_solution = init_solution
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.neigh_structure = neigh_structure

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
            max_iter = (max_eval) // len(domain)
        maxm = -1 * math.inf
        minm = math.inf
        neigh = {}
        for i in range(max_iter):
            for dom in domain:
                neigh[dom] = np.random.choice(domain[dom])
            val = self.run(neigh, neigh, X, y)
            minm = min(minm, val)
            maxm = max(maxm, val)

        return (minm, maxm)



    def Mf(self, k: int) -> int:
        """The method for represtenting the maximum number of failures allowed while iterating for the kth step in the Stochastic Ruler Method
            In this case, we let Mk = floor(log_5 (k + 10)) for all k; this choice of the sequence {Mk} satisfies the guidelines specified by Yan and Mukai (1992)
        Args:
            k (int): the iteration number in the Stochastic Ruler Method

        Returns:
            int: the maximum number for the number of iterations
        """

        return int(math.log(k + 10, math.e) / math.log(5, math.e))


    


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

        if self.percentReduction is not None:

            initial_choice_HP = {}

            if self.init_solution is None:
                for i in self.space:
                    initial_choice_HP[i] = self.space[i][0]
            else:
                initial_choice_HP = self.init_solution

            # printing initial value for checking
            init_value = self.func(initial_choice_HP)
            print("Initial value = ", init_value)

            self.target_value = init_value * (1 - self.percentReduction * 0.01) if init_value >= 0 else init_value * (1 + self.percentReduction * 0.01)
            print("Target Value = ", self.target_value)
            print("------")

            # step 0: Select a starting point x0 in S and let k = 0
            k = 1
            x_k = initial_choice_HP
            opt_x = x_k
            a, b = self.det_a_b(self.space, self.maxevals // 10, X, y)
            # print(a,b)
            minh_of_z = b
            # step 0 ends here
            while k < self.maxevals + 1:
                # print("minz"+str(minh_of_z))
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
                    print("value at iter: ", h_of_z)
                    

                    if h_of_z <= self.target_value :
                        print("-------")
                        print("target val:", self.target_value)
                        print("h of z at stop: ", h_of_z)

                        print("Stopping criterion of ", self.percentReduction,"% reduction in function value. Stopping optimization.")
                        # return h_of_z, opt_x, a, b, minh_of_z_tracker
                        self.minh_of_z_tracker.append(h_of_z)
                        print(self.minh_of_z_tracker)
                        #print(target_value)
                        return h_of_z, opt_x, a, b

                    u = np.random.uniform(a, b)  # Then draw a sample u from U(a, b)

                    if h_of_z > u:  # If h(z) > u, then let xk+1 = xk and go to step 3.
                        k += 1
                        if h_of_z < minh_of_z:
                            minh_of_z = h_of_z
                            opt_x = x_k
                        break
                    # Otherwise draw another sample h(z) from H(z) and draw another sample u from U(a, b), part of the loop where iter = self.Mf(k) tells the maximum number of failures allowed
                    if h_of_z < u:  # If all Mk tests have failed
                        x_k = zk
                        k += 1
                        if h_of_z < minh_of_z:
                            minh_of_z = h_of_z
                            self.minh_of_z_tracker.append(h_of_z)
                            opt_x = zk
                    
                # step 2 ends here
                # step 3: k = k+1

            # return minh_of_z, opt_x, a, b, minh_of_z_tracker
            return minh_of_z, opt_x, a, b 

        else:
            print("No percentRedn criteria set:")

            initial_choice_HP = {}

            if self.init_solution is None:
                for i in self.space:
                    initial_choice_HP[i] = self.space[i][0]
            else:
                initial_choice_HP = self.init_solution

            # printing initial value for checking
            init_value = self.func(initial_choice_HP)
            print(init_value)

            # step 0: Select a starting point x0 in S and let k = 0
            k = 1
            x_k = initial_choice_HP
            opt_x = x_k
            a, b = self.det_a_b(self.space, self.maxevals // 10, X, y)
            # print(a,b)
            minh_of_z = b
            # step 0 ends here
            while k < self.maxevals + 1:
                # print("minz"+str(minh_of_z))
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
                    print(h_of_z)
                    

                    u = np.random.uniform(a, b)  # Then draw a sample u from U(a, b)

                    if h_of_z > u:  # If h(z) > u, then let xk+1 = xk and go to step 3.
                        k += 1
                        if h_of_z < minh_of_z:
                            minh_of_z = h_of_z
                            opt_x = x_k
                        break
                    # Otherwise draw another sample h(z) from H(z) and draw another sample u from U(a, b), part of the loop where iter = self.Mf(k) tells the maximum number of failures allowed
                    if h_of_z < u:  # If all Mk tests have failed
                        x_k = zk
                        k += 1
                        if h_of_z < minh_of_z:
                            minh_of_z = h_of_z
                            self.minh_of_z_tracker.append(minh_of_z)
                            opt_x = zk

                # step 2 ends here
                # step 3: k = k+1

            # return minh_of_z, opt_x, a, b, minh_of_z_tracker
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

            # print(funcval,opt_x)
            return funcval
        # print("acc" + str(acc))

    def plot_minh_of_z(self):
        """Plot the variation of minh_of_z with each iteration."""
        # _, _, _, _, minh_of_z_tracker = self.SR_Algo()  Call the modified SR_Algo method

        plt.figure(figsize=(10, 6))

        if self.percentReduction is not None:
            plt.axhline(y=self.target_value, color='r', linestyle='--')

        plt.plot(self.minh_of_z_tracker, marker='o', linestyle='-')
        plt.xlabel('Iteration k')
        plt.ylabel('minh_of_z')
        plt.title('Variation of Objective Function Value with Each Iteration')
        plt.grid(True)
        plt.show()    



