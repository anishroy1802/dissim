import numpy as np
import math
import itertools
import pandas as pd
from mrg32k3a.mrg32k3a import MRG32k3a

class KN():
    def __init__(self, mu, sigma, domain, step_size, func, alpha, delta, crn=False, n_0=2, max_evals=300, print_solutions=True):
        self.mu = mu
        self.sigma = sigma
        self.domain = domain
        self.step_size = step_size
        self.func = func
        self.alpha = alpha
        self.delta = delta
        self.crn = crn
        self.max_evals = max_evals
        self.n_0 = n_0
        self.dimensions = len(self.domain)
        self.print_solutions = print_solutions
        sol_space = []
        for i in range(self.dimensions):
            sol_space.append(np.arange(self.domain[i][0], self.domain[i][1] + step_size[i], step_size[i]))

        self.solution_space = list(itertools.product(*sol_space))
        self.confidence_lvl = 1 - self.alpha
        self.eta = 0.5 * (((2 * self.alpha) / (len(self.solution_space) - 1)) ** (-2 / (self.n_0 - 1)))

        if self.crn:
            # Initialize MRG32k3a generator with a seed- do i keep this as a parameter?
            self.rng = MRG32k3a(s_ss_sss_index=[1, 2, 3])  # You can provide a seed as a list of three integers
        else:
            self.rng = None


    def generate_noise(self, mu, sigma):
        if self.crn:
            return self.rng.normalvariate(mu= mu, sigma= sigma)   # Use rand() method to generate random numbers between 0 and 1
        else:
            return np.random.normal(loc = mu, scale= sigma)
        
    def reset_rng_substream(self):
        if self.rng is not None and self.crn is True:
            self.rng.reset_substream()

    def advance_rng_substream(self):
        if self.rng is not None and self.crn is True:
            self.rng.advance_substream()

    def initialize(self):
        a = len(self.solution_space)
        self.h = math.sqrt(2 * self.eta * (self.n_0 - 1))
        self.sol_space_dict = {i: self.solution_space[i] for i in range(len(self.solution_space))}
        self.X_i_bar = [0] * len(self.solution_space)
        self.X_i_bar_n_0 = [0] * len(self.solution_space)
        self.S_sq = np.zeros((a, a))
        S_sq = np.zeros((a, a))
        self.sim_vals = np.zeros((a, self.n_0)).tolist()
        self.reset_rng_substream()
        for i in range(len(self.solution_space)):
            for simrep in range(0, self.n_0):
                noise = self.generate_noise(mu=self.mu, sigma=self.sigma)
                self.sim_vals[i][simrep] = self.func(self.sol_space_dict[i], noise)
                #print("noise ", noise, "i: ", i, "simrep: ", simrep)
                self.X_i_bar_n_0[i] += self.sim_vals[i][simrep]

            if self.crn == True:
                if i!= len(self.solution_space) - 1:
                    self.reset_rng_substream()
                else:
                    self.advance_rng_substream()
            
            self.X_i_bar[i] = sum(self.sim_vals[i])
            self.X_i_bar[i] = self.X_i_bar[i] / len(self.sim_vals[i])
            self.X_i_bar_n_0[i] = self.X_i_bar_n_0[i] / self.n_0
        
        for i in range(a):
            for j in range(a):
                S_sq[i][j] = (1 / (self.n_0 - 1)) * (sum(self.sim_vals[i][0:self.n_0])
                                                      - sum(self.sim_vals[j][0:self.n_0])
                                                      - (self.X_i_bar_n_0[i] - self.X_i_bar_n_0[j]))**2
        
        return S_sq

    def optimize(self):
        print("CRN used: ", self.crn)
        fx_values = []
        x_values = []
        if self.max_evals < 200:
            print("Too small budget")
            return 

        self.S_sq = self.initialize()
        r = self.n_0
        a = len(self.solution_space)
        I_old = set()
        
        for i in range(0, a):
            I_old.add(i)

        while len(I_old) != 1 and self.max_evals > 0:
            I = set()
            self.check = np.zeros((a, a))
            self.W = np.zeros((len(self.solution_space), len(self.solution_space)))
            for i in range(len(self.solution_space)):
                for j in range(len(self.solution_space)):
                    b = (self.delta / (2 * r)) * (((self.h ** 2 * self.S_sq[i][j]) / (self.delta ** 2)) - r)
                    self.W[i][j] = max(0, b)

            for i in range(0, a):
                for j in range(0, a):
                    if ((i != j) and (i in I_old) and (j in I_old)):
                        if self.X_i_bar[i] >= self.X_i_bar[j] - self.W[i][j]:
                            self.check[i][j] = 1

                row_sum = sum(self.check[i])
                if row_sum == len(I_old) - 1:
                    I.add(i)
            
            if len(I) == 1:
                print("Got single optimal solution: ")
                for ele in I:
                    x_values.append(self.sol_space_dict[ele])
                    fx_values.append(-1*self.X_i_bar[ele])
                self.df = pd.DataFrame({'x*': x_values, 'f(x*)': fx_values})
                if self.print_solutions:
                    print(self.df)
                return I
            else:
                I_old = I.copy()
                r += 1
                self.advance_rng_substream()
                #r = max(r, 3* self.n_0)
                for i in range(a):
                    if i in I_old:
                        noise = self.generate_noise(mu=self.mu, sigma=self.sigma)  # Generate noise
                        #print("noise ", noise, "i: ", self.sol_space_dict[i], "simrep: ", r)
                        self.sim_vals[i].append(self.func(self.sol_space_dict[i], noise))
                        self.max_evals -= 1
                        self.X_i_bar[i] = (self.X_i_bar[i]*(r-1) + self.sim_vals[i][-1])/r

                    self.reset_rng_substream()

        print("Final set after exhausting budget: ")   

        if I is not None:
            for ele in I:
                x_values.append(self.sol_space_dict[ele])
                fx_values.append(-1*self.X_i_bar[ele])
            self.df = pd.DataFrame({'x*': x_values, 'f(x*)': fx_values})

            if self.print_solutions:
                print(self.df)
        return I

# Define the objective function
def objective_function(x, noise):
    return -1 * (2*x[0] + x[0]**2 + x[1]**2 + noise)

# Main code
dom = [[0, 2], [0, 2]]
step_size = [0.5, 0.5]

# Initialize and run the optimization without control variates
optimizer = KN(domain=dom, step_size=step_size, func=objective_function,
               alpha=0.5, delta=5, n_0=2, max_evals=500, mu = 0.3, sigma = 0.1, crn=False)
a1 = optimizer.optimize()

# Initialize and run the optimization with control variates
optimizer_with_crn = KN(domain=dom, step_size=step_size, func=objective_function,
                        alpha=0.5, delta=5, n_0=2, max_evals=500, mu = 0.3, sigma= 0.1, crn=True)
a2 = optimizer_with_crn.optimize()
