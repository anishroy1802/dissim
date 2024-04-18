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


    def min_list(self, lst):
        if not lst:
            return None  # Return None if the list is empty
        min_val = lst[0]  # Initialize min_val with the first element of the list
        for num in lst[1:]:  # Iterate through the list starting from the second element
            if num < min_val:  # If the current number is smaller than min_val
                min_val = num  # Update min_val with the current number
        return min_val  # Return the minimum value
    
    
    def update_dictionary(self, dict1, dict2, avg_value):
        # Convert values of the first dictionary into a tuple
        key_tuple = tuple(dict1.values())
        
        # Check if the key tuple already exists in the second dictionary
        if key_tuple in dict2:
            # If it exists, update the value with the mean of avg_value and existing value
            dict2[key_tuple] = (dict2[key_tuple] + avg_value) / 2
        else:
            # If it doesn't exist, add the key tuple to the second dictionary with avg_value as the value
            dict2[key_tuple] = avg_value
    
    def find_key_by_value(self, number, dictionary):
        for key, value in dictionary.items():
            if value == number:
                return key
        return None  # Return None if the number is not found in the dictionary


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
        self.avg_value_tracker = []
        solutions_visited = {}

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
            minh_of_z = b
            # step 0 ends here
            while k < self.max_evals:
                
                # step 1:  Given xk = x, choose a candidate zk from N(x)
                if self.neigh_structure == 1:
                    zk = self.next_solution_based_on_distance(x_k)                  #N1
                
                elif self.neigh_structure == 2:
                    zk = self.random_pick_from_neighbourhood_structure(x_k)         #N2


                # step 1 ends here
                # step 2: Given zk = z, draw a sample h(z) from H(z)
                iter = self.Mf(k)
                index = 0
                # step 0 ends here
                while k < self.max_evals:

                    f_avg = 0
                    i=0

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
                        k += 1

                        if f_avg/(i+1) <= self.target_value :
                            # print("h of z at stop: ", h_of_z)

                            print("Stopping criterion of ", self.percent_improvement,"% reduction in function value. Stopping optimization.")
                            self.minh_of_z_tracker.append(h_of_z)
                            self.avg_value_tracker.append(f_avg/(i+1))
                            minima = self.min_list(self.avg_value_tracker)
                            return minima, opt_x, a, b

                        u = np.random.uniform(a, b)  # Then draw a sample u from U(a, b)

                        if h_of_z > u:
                            # print("iters", iter, "sol no: ", index, "z_k = ", zk,  "Not optimal, SR algo reject")
                            break

                        if h_of_z < u:  # If all Mk tests have failed
                            x_k = zk
                            # f_avg+=h_of_z
                            if h_of_z < minh_of_z:
                                # f_avg+=h_of_z
                                opt_x = zk
                                minh_of_z = h_of_z
                                self.minh_of_z_tracker.append(minh_of_z)
                                
                        #updating f_of_z outside seems more accurate. THINK. 
                    # minima = self.min_list(self.avg_value_tracker)
                    index +=1
                    if (f_avg != 0):
                        self.avg_value_tracker.append(f_avg/iter)
                    if self.print_solutions:
                        print("iters = ", iter, "sol no: ", index, "z_k = ", zk, "avg value at zk = ", f_avg/iter)                    



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
                    
                    k += 1

                    u = np.random.uniform(a, b)  # Then draw a sample u from U(a, b)

                    if h_of_z > u:
                        # print("iters", iter, "sol no: ", index, "z_k = ", zk,  "Not optimal, SR algo reject")
                        break

                    if h_of_z < u:  # If all Mk tests have failed
                        x_k = zk
                        # f_avg+=h_of_z
                        if h_of_z < minh_of_z:
                            # f_avg+=h_of_z
                            opt_x = zk
                            minh_of_z = h_of_z
                            self.minh_of_z_tracker.append(minh_of_z)
                            
                #updating f_of_z outside seems more accurate. THINK. 
                index +=1
                if (f_avg != 0):
                    self.avg_value_tracker.append(f_avg/iter)
                    if self.print_solutions:
                        # print('k = ', k, "iters = ", iter, "sol no: ", index, "z_k = ", zk, "avg value at zk = ", f_avg/iter) # k gives an idea of the number of replications before rejection
                        print( "iters = ", iter, "sol no: ", index, "z_k = ", zk, "avg value at zk = ", f_avg/iter)
                        if k >= self.max_evals:
                            print("Solutions visited and overall average value = ", solutions_visited)

                self.update_dictionary (zk, solutions_visited, f_avg/iter )
                somelist = list(solutions_visited.values())
                minima = self.min_list(somelist)
                opt_x = self.find_key_by_value(minima, solutions_visited)
                       

            # print("single rep minimum tracker: ", self.minh_of_z_tracker)
            # print("average value tracker: ", self.avg_value_tracker)
            # print("minimum average value:", minima)
            # print ("dictionary of values = ", solutions_visited)
            # print("length of dictionary = ", len(solutions_visited))
            # print("length of avg = " , len(self.avg_value_tracker))

            return minima, opt_x, a, b

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



