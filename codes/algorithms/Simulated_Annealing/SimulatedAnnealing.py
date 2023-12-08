"""" Package Imports:
In this section, we import several essential Python libraries that will be used throughout the code.
 - 'numpy' (imported as 'np') is a fundamental library for numerical operations, providing support for arrays and mathematical functions.
 - 'scipy.spatial.distance' imports the 'euclidean' function, which calculates the Euclidean distance between points.
 - 'tracemalloc' is used to monitor memory usage and allocation during code execution.
 - 'matplotlib.pyplot' (imported as 'plt') allows for creating various plots and visualizations.
 - 'math' provides a set of mathematical functions and constants for use in calculations."""
import matlab.engine
import numpy as np
from scipy.spatial.distance import euclidean
import tracemalloc
import matplotlib.pyplot as plt
import math
import itertools
import random

#Get the folder path where filename is located
matlab_script_folder = r'C:\Users\hp\dissim\codes\algorithms\Simulated_Annealing\examples'

# Initialize the MATLAB Engine
# eng = matlab.engine.start_matlab()
# # Add the folder containing 'e4.m' to the MATLAB path
# eng.addpath(matlab_script_folder)

"""Memory Tracing:
 The 'tracing_start' function is defined to manage memory tracing. It begins by stopping any ongoing memory tracing using 'tracemalloc.stop()' and checking if tracing is currently active with 'tracemalloc.is_tracing()'. Afterward, it starts memory tracing with 'tracemalloc.start()' and checks the tracing status again. This function is invaluable for profiling memory usage, enabling developers to gain insights into memory allocation and deallocation patterns during code execution. It can help identify potential memory-related issues and optimize memory usage in the application."""

def tracing_start():
    tracemalloc.stop()
    print("Tracing Status: ", tracemalloc.is_tracing())
    tracemalloc.start()
    print("Tracing Status: ", tracemalloc.is_tracing())

# def matlab_to_python_wrapper(func_name,x): 
#     result = eng.func_name(matlab.double(x), nargout=1)
#     noise = np.random.normal(0, 0.3)  # Mean = 0, Standard Deviation = 0.3
#     result_with_noise = float(result) + noise
#     return result_with_noise

class SA():

    def __init__(self, dom, step_size, T, custom_H_function, nbd_structure, k= 300, platform = "python", percent_reduction=None, initial_solution=None, random_seed=None):

        self.platform = platform #platform can be "python" or "matlab"
        self.custom_H_function = custom_H_function
        self.filename = self.custom_H_function.__name__
        self.domain = dom #n*2 matrix
        self.dimensions = len(self.domain)
        print("Number of dimensions:", self.dimensions)
        self.step_size = step_size #n*1 matrix
        self.T = T
        self.k = k
        self.L = [i + 200 for i in range(0, k)]
        self.initial_solution = initial_solution
        self.random_seed = random_seed
        self.X_opt = []
        self.nbd_structure = nbd_structure
        self.function_values = []
        self.initial_function_value = None  # Store the initial function value
        self.percent_reduction = percent_reduction
        self.max_euclidean_value = math.sqrt(sum(element[1]**2 - element[0]**2 for element in self.domain))
        self.history = []
        

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
                    self.R_prime[element][neighbor] = self.max_euclidean_value - euclidean(element, neighbor)
                    self.D[element] += self.R_prime[element][neighbor]
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
        if self.initial_solution is None:
            if self.random_seed is None:
                random.seed(1234)
            else:
                random.seed(self.random_seed)
            random_index = random.randint(0, len(self.solution_space) - 1)
            X0 = position_index[random_index]
        else:
            X0 = self.initial_solution
        print("initial solution:", X0)

        self.V={}
        self.V[X0] = 1
        self.X_opt.append(X0)

        self.function_values.append(self.custom_H_function(list(X0)))
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

            if j == 0:
                self.history.append(fx)
            G_xz = np.exp(-(fz - fx) / self.T)
            if np.random.rand() <= G_xz:
                x_next = z_j 
                self.history.append(fz)
            else :
                x_next = x_j
                self.history.append(fx)

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
        print("iters:", len(self.function_values)-1)
        # result_string = ' '.join(map(str, self.X_opt))
        # print("values: ", result_string)
        #print("VALUES:")
        # print("init:", self.function_values[0])
        # print("final:", self.function_values[-1])
        print("final optimal value: ", self.X_opt[-1])
        print("% reduction:",  100*(self.function_values[0] -  self.function_values[-1])/abs(self.function_values[0]))
        print(self.X_opt)

def objective_function(x):
    noise = np.random.normal(scale=0.1)  # Add Gaussian noise with a standard deviation of 0.1
    return 2*x[0] + x[0]**2 + x[1]**2 + noise

def func(x0):
  x1,x2 = x0[0],x0[1]
  def multinodal(x):
    return (np.sin(0.05*np.pi*x)**6)/2**(2*((x-10)/80)**2)

  return -(multinodal(x1)+multinodal(x2))+np.random.normal(0,0.3)

def func2(x):
  x1,x2,x3,x4 = x[0],x[1], x[2],x[3]
  return (x1+10*x2)**2 + 5* (x3-x4)**2 + (x2-2*x3)**4 + 10*(x1-x4)**4 
  + 1 +np.random.normal(0,30)
#main()

def facility_loc(x):
  #normal demand
  X1,Y1 = x[0],x[1] 
  X2,Y2 = x[2],x[3] 
  X3,Y3 = x[4],x[5] 
  avg_dist_daywise = []
  T0 = 30
  n = 6
  for t in range(T0):
      total_day = 0                  ##### total distance travelled by people
      ###### now finding nearest facility and saving total distance 
      #travelled in each entry of data
      for i in range(n):
          for j in range(n):
              demand=-1
              while(demand<0):    
                  demand = np.random.normal(180, 30, size=1)[0]
              total_day += demand*min(abs(X1-i)+abs(Y1-j) ,
                                      abs(X2-i)+abs(Y2-j),abs(X3-i)+abs(Y3-j) ) 
              ### total distance from i,j th location to nearest facility
      avg_dist_daywise.append(total_day/(n*n))    
  return sum(avg_dist_daywise)/T0


dom = [[1,4]]*6
step_size = [1]*len(dom)
T= 100
k= 100


optimizer  = SA(dom = dom, step_size= step_size, T = 100, k = 50,
                         custom_H_function= facility_loc, nbd_structure= 'N1', 
                         random_seed= 42, percent_reduction=40)
optimizer.optimize()
optimizer.print_function_values()


optimizer  = SA(dom = dom, step_size= step_size, T = 100, k = 50,
                         custom_H_function= facility_loc, nbd_structure= 'N2', 
                         random_seed= 42, percent_reduction=40)
optimizer.optimize()
optimizer.print_function_values()
# optimizer  = SA(dom = dom, step_size= step_size, T = 100, k = 100,
#                          custom_H_function= facility_loc, nbd_structure= 'N2', 
#                          random_seed= 42, percent_reduction=60, platform= 'matlab')
# optimizer.optimize()
#optimizer.print_function_values()

# Example usage
optimizer = SA(domain_min=-1, domain_max=1, step_size=0.1, T=100,  custom_H_function=H, nbd_structure="N2", percent_reduction=10)
optimizer.optimize()
optimizer.print_function_values()