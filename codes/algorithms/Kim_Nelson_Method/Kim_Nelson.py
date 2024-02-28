import numpy as np

def static_truss_4bar_varun(para, n):
    # objective to minimize expectation of hfun
    # design variable = para,  para_lower=0;   para_upper=175;
    # n= number of realizations of the function to estimate expectation
    # output function evaluations size (n,1)
    theta = np.random.randn(n, 3)  # three is the number of random variables
    
    len_val = 1000  # member length
    Area1 = para
    Area2 = 250 - np.sqrt(2) * Area1
    Coord = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [2, 0, 0],
                      [1, -1, 0],
                      [2, -1, 0]]) * len_val
    Con = np.array([[1, 4],
                    [2, 4],
                    [3, 4],
                    [5, 4]])  # Corrected indices
    Reaction = np.array([[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1],
                         [0, 0, 1],
                         [1,1,1]])
    Area = np.array([Area1, Area2, Area1, Area2])
    Rho = np.ones(Con.shape[0]) * 7860

    hfun = np.zeros((n, 1))

    for kk in range(n):
        Load = np.array([[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0],
                         [100 + 20 * theta[kk, 0], 0, 0],
                         [0, 0, 0]])
        Elasticity1 = 200 + 30 * theta[kk, 1]
        Elasticity2 = 80 + 10 * theta[kk, 2]
        Elasticity = np.array([Elasticity1, Elasticity2, Elasticity1, Elasticity2])
        D = {'Coord': Coord.T, 'Con': Con.T, 'Re': Reaction.T, 'Load': Load.T, 'E': Elasticity, 'A': Area, 'R': Rho}
        for key, value in D.items():
            print(f'Shape of {key}: {value.shape}')
        w = D['Re'].shape
        # Global Stiffness Matrix S
        S = np.zeros((3 * w[1], 3 * w[1]))
        # Global Mass Matrix M
        M = np.zeros((3 * w[1], 3 * w[1]))
        # Unrestrained Nodes U
        U = np.ones(w) - D['Re']
        # Location of unrestrained nodes f
        f = np.where(U)[0]

        Tj = np.zeros((3 * D['Con'].shape[1], D['Con'].shape[1]))  # Initialize Tj

        for i in range(D['Con'].shape[1]):
            H = D['Con'][:, i]
            C = D['Coord'][:, H[1]] - D['Coord'][:, H[0]]
            print("Works 1")
            # Length of Element Le
            Le = np.linalg.norm(C)
            T = C / Le
            s = np.outer(T, T)
            e = np.concatenate(([3 * H[0] - 2, 3 * H[0], 3 * H[0] + 1], [3 * H[1] - 2, 3 * H[1], 3 * H[1] + 1]))
            print("Works 2")
            # Stiffness for element i G=EA/Le
            G = D['E'] * D['A'] / Le
            print(G.shape)
            print("Works 3")
            print(S[np.ix_(e, e)].shape, G.shape, np.array([[s, -s], [-s, s]]).shape)
            S[np.ix_(e, e)] += np.dot(G ,np.array([[s, -s], [-s, s]]))
            print("Works 4")
            # Mass of element i
            K = (D['A'][i] * D['R'][i] * Le / 2)
            M[np.ix_(e, e)] += K * np.block([[np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), np.eye(3)]])
            Tj[e, i] = G * T  # Store Tj for each element
        U[f] = np.linalg.solve(S[np.ix_(f, f)], D['Load'][f])  # node displacements
        F = np.sum(np.multiply(Tj, (U[:, D['Con'][1, :]] - U[:, D['Con'][0, :]])))  # member forces
        R = (S @ U.flatten()).reshape(w)  # reactions
        Stiff = S[np.ix_(f, f)]  # stiffness matrix
        Mass = M[np.ix_(f, f)]  # mass matrix
        hfun[kk] = U[0, 3]

    return hfun


np.random.seed(42)
alpha = 0.05
delta = 0.2
n0 = 500
k = 174
eta = 0.5 * ((2 * alpha / (k - 1)) ** (-2 / (n0 - 1)) - 1)
hsq = 2 * eta * (n0 - 1)
h = np.sqrt(hsq)
reps = np.zeros((n0, k))
for i in range(k):
    np.random.seed(100 + i)
    reps[:, i] = static_truss_4bar_varun(i, n0)

sys_mean = np.zeros((k, 1))
sys_std = np.zeros((k, k))
w = np.zeros((k, k))
for i in range(k):
    sys_mean[i, 0] = np.mean(reps[:, i])

for i in range(k):
    for j in range(k):
        if i == j:
            sys_std[i, j] = 0
            w[i, j] = 0
        else:
            s = 0
            for b in range(n0):
                s += (reps[b, i] - reps[b, j] - (sys_mean[i] - sys_mean[j])) ** 2
            sys_std[i, j] = (1 / (n0 - 1)) * s
            w[i, j] = max(0, (delta / (2 * n0)) * ((hsq * sys_std[i, j] / (delta ** 2)) - n0))

I = np.zeros((k, 1))
for i in range(k):
    a = 0
    for j in range(k):
        if i == j:
            a = a
        else:
            if sys_mean[i] - sys_mean[j] - w[i, j] > 0:
                a += 1
    if a == 0:
        I[i, 0] = i
    else:
        I[i, 0] = 0

print(I)

