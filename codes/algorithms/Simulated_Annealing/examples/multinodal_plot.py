import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the multinodal function
def multinodal(x):
    return (np.sin(0.05 * np.pi * x) ** 6) / 2 ** (2 * ((x - 10) / 80) ** 2)

# Define the func1 function
def func1(x0):
    x1, x2 = x0[0], x0[1]
    return -(multinodal(x1) + multinodal(x2)) + np.random.normal(0, 0.3)

# Define the domains for x1 and x2
x1_domain = np.linspace(0, 100, 100)  # Adjust the number of points as needed
x2_domain = np.linspace(0, 100, 100)  # Adjust the number of points as needed

# Create a grid of (x1, x2) values
x1_grid, x2_grid = np.meshgrid(x1_domain, x2_domain)

# Calculate the corresponding values of func1 for each (x1, x2) pair
func1_values = np.zeros_like(x1_grid)
for i in range(len(x1_domain)):
    for j in range(len(x2_domain)):
        x1 = x1_grid[i, j]
        x2 = x2_grid[i, j]
        func1_values[i, j] = func1([x1, x2])

# Create a 3D surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_grid, x2_grid, func1_values, cmap='viridis')

# Set labels for the axes
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('func1 Value')

# Show the plot
plt.show()
