import numpy as np
import dissim

# Define cities and their coordinates (on a discrete map with step distance of 0.5)
num_cities = 10
cities = np.random.randint(0, 20, size=(num_cities, 2)) * 0.5  # Discrete coordinates for each city with step distance of 0.5 units

# Objective function: Total distance traveled in the TSP
def tsp_distance(route):
    total_distance = 0
    for i in range(num_cities - 1):
        city1 = route[i]
        city2 = route[i + 1]
        # Calculate the base distance between cities
        base_distance = np.linalg.norm(cities[city1] - cities[city2])
        # Introduce randomness to simulate variations such as road conditions or traffic congestion
        random_perturbation = np.random.normal(0, 1)  # Adjust the parameters as needed
        # Add the perturbed distance to the total distance
        total_distance += base_distance + random_perturbation
    # Add distance from last city back to the starting city
    total_distance += np.linalg.norm(cities[route[-1]] - cities[route[0]])
    return total_distance

# Define the domain for the TSP (permutation of cities)
tsp_domain = [(0, num_cities - 1)] * num_cities  # Each city is represented by its index

# Initial guess for the route (random permutation)
initial_route = np.random.permutation(num_cities)

# Instantiate the AHA optimizer
optimizer_tsp = dissim.AHA(tsp_distance, tsp_domain, percent=80)

# Run the optimization algorithm
result_tsp = optimizer_tsp.AHAalgolocal(100, tsp_domain, initial_route, 4000)

# Optimal route and its total distance
optimal_route = result_tsp[-1]
optimal_distance = tsp_distance(optimal_route)

# Print coordinates of each city
print("Coordinates of each city:")
for i, city_coord in enumerate(cities):
    print(f"City {i+1}: {city_coord}")

# Print the optimal route and its total distance
print("\nOptimal solution:")
print("Optimal route:", optimal_route)
# print("Total distance of optimal route:", optimal_distance)

# Plot the iterations during the optimization process
optimizer_tsp.plot_iterations()

# Print problem parameters
print("\nParameters of the Traveling Salesman Problem:")
print("Number of cities:", num_cities)
print("Initial route (random permutation):", initial_route)
