"""Implementation of the Random Search algorithm for the optimization of a function f(x)"""
from typing import Any, Callable
import numpy as np

def basin_func_1(x: tuple, a: float = 0.5, h: float = 2, k: float = -5) -> float:
    """
    :param x: Each element of the tuple is an x-coordinate of a point in the plane
    :param a: The coefficient of the quadratic term. Defaults to 0.5
    :param h: The x-coordinate of the vertex. Defaults to 2
    :param k: The y-coordinate of the vertex. Defaults to -5
    :return: The value of the function at x
    """
    return sum(a * (xi - h)**2 + k for xi in x)

def basin_func_2(x: tuple) -> float:
    """
    :param x: Each element of the tuple is an x-coordinate of a point in the plane
    :return: The value of the function at x
    """
    return sum(xi**2 for xi in x)

def random_search(basin_function: Callable, goal: str, x_range: tuple[float, float], n: int = 2, kwargs: dict[str, Any] = {}, max_iterations: int = 10000):
    """
    Performs a random search optimization to find the minimum or maximum of the given basin function.
    
    :param basin_function: The basin function to optimize
    :param goal: 'min' for minimization, 'max' for maximization
    :param x_range: The range (min, max) of values for each element of x
    :param n: Problem size. Defaults to 2
    :param args: Additional arguments to pass to the basin function. Defaults to {}
    :param max_iterations: The maximum number of iterations to perform. Defaults to 10000
    :return: The best solution found and its cost
    """
    solution = None
    y = float('inf') if goal == 'min' else -float('inf')
    
    for _ in range(max_iterations):
        candidate_solution = np.random.uniform(x_range[0], x_range[1], n)
        candidate_cost = basin_function(candidate_solution, **kwargs)
        
        if (goal == 'min' and candidate_cost < y) or (goal == 'max' and candidate_cost > y):
            y = candidate_cost
            solution = candidate_solution
    
    return solution, y


if __name__ == "__main__":
    x_range = (-5, 5)
    solution, y = random_search(basin_func_1, 'min', x_range)
    print(f'Best solution found: {solution}')
    print(f'Cost: {y}')
    

    x_range = (-5, 5)
    solution, y = random_search(basin_func_2, 'min', x_range)
    print(f'Best solution found: {solution}')
    print(f'Cost: {y}')