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

        # else:
        #     max_iter = (max_eval) // len(domain)
        # maxm = -1 * math.inf
        # minm = math.inf
        # neigh = {}
        # for i in range(max_iter):
        #     for dom in domain:
        #         neigh[dom] = np.random.choice(domain[dom])
        #     val = self.run(neigh, neigh, X, y)
        #     minm = min(minm, val)
        #     maxm = max(maxm, val)

        # return (minm, maxm)
    
        # if self.lower_bound is not None and self.upper_bound is not None:
        #     minm = self.lower_bound
        #     maxm = self.upper_bound
        else:
            max_iter = 20
        
            Reps_for_each_sol = 5
            maxm = -math.inf
            minm = math.inf

            for i in range(max_iter):
                print("i = ", i)
                for j in range(Reps_for_each_sol):
                    print("j = ", j)
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
            # a = -2
            # b = 1
            print("a,b:")
            print(a,b)
            minh_of_z = b
            # step 0 ends here
            print("total evals: ", self.maxevals + 1)
            while k < self.maxevals + 1:
                print("k: " , k)
                print("x_k: ", x_k)
                k+=1

                # print("minz"+str(minh_of_z))
                # step 1:  Given xk = x, choose a candidate zk from N(x)

                if self.neigh_structure == 1:
                    zk = self.next_solution_based_on_distance(x_k)                  #N1
                
                elif self.neigh_structure == 2:
                    zk = self.random_pick_from_neighbourhood_structure(x_k)         #N2


                # step 1 ends here
                # step 2: Given zk = z, draw a sample h(z) from H(z)
                print("z_K: ", zk)
                iter = self.Mf(k)
                print("iter: ", iter)
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
                        #k += 1
                        if h_of_z < minh_of_z:
                            minh_of_z = h_of_z
                            opt_x = x_k
                        break
                    # Otherwise draw another sample h(z) from H(z) and draw another sample u from U(a, b), part of the loop where iter = self.Mf(k) tells the maximum number of failures allowed
                    if h_of_z <= u:  # If all Mk tests have failed
                        x_k = zk
                        #k += 1
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
            # a = -2
            # b= 1
            print(a,b)
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

    # def plot_minh_of_z(self):
    #     """Plot the variation of minh_of_z with each iteration."""
    #     # _, _, _, _, minh_of_z_tracker = self.SR_Algo()  Call the modified SR_Algo method

    #     plt.figure(figsize=(10, 6))

    #     if self.percentReduction is not None:
    #         plt.axhline(y=self.target_value, color='r', linestyle='--')

    #     plt.plot(self.minh_of_z_tracker, marker='o', linestyle='-')
    #     plt.xlabel('Iteration k')
    #     plt.ylabel('minh_of_z')
    #     plt.title('Variation of Objective Function Value with Each Iteration')
    #     plt.grid(True)
    #     plt.show()


# def func(x0):
#     x1, x2 = x0["x1"], x0["x2"]

#     def multinodal(x):
#         return (np.sin(0.05 * np.pi * x) ** 6) / 2 ** (2 * ((x - 10) / 80) ** 2)

#     return -(multinodal(x1) + multinodal(x2)) + np.random.normal(0, 0.3)


# dom = {"x1": [i for i in range(101)], "x2": [i for i in range(101)]}

# sr_userDef = stochastic_ruler(space=dom, maxevals=10, prob_type="opt_sol", func=func, neigh_structure=1)
# print(sr_userDef.optsol())
# #sr_userDef.plot_minh_of_z()

def func2(x):
  x1,x2,x3,x4 = x['x1'],x['x2'], x['x3'],x['x4']
  return (x1+10*x2)**2 + 5* (x3-x4)**2 + (x2-2*x3)**4 + 10*(x1-x4)**4 + 1 +np.random.normal(0,30)

dom3 = {'x1':[i for i in range(20)],'x2':[i for i in range(20)],'x3':
       [i for i in range(20)],'x4':[i for i in range(20)]}
sr_userDef3 = stochastic_ruler(space = dom3, maxevals = 10, prob_type = 'opt_sol', func = func2, percentReduction = 50, neigh_structure=1)
print(sr_userDef3.optsol())
# sr_userDef3.plot_minh_of_z()