import numpy as np

from scipy.special import eval_genlaguerre
from typing import Union, List

def generalized_laguerre(l: int, p: int, x_array: Union[np.ndarray, List[float]]) -> np.ndarray:
    """
    Evaluate the generalized Laguerre polynomial L_{p}^{|l|}(x)
    for the given set (l, p) and array of real numbers x_array.

    Parameters
    ----------
    l : int
        Degree of associatedness.
    p : int
        Degree of polynomial.
    x_array : array-like
        Real numbers for which to evaluate the polynomial.

    Returns
    -------
    np.array
        Evaluated polynomial values.
    """
    # Convert x_array to numpy array for vectorized operations
    x_array = np.array(x_array)
    
    # Evaluate the generalized Laguerre polynomial
    result = eval_genlaguerre(p, abs(l), x_array)
    
    return result


def hermite_physics(n: int, x_array: Union[np.ndarray, List[float]]) -> np.ndarray:
    """
    Evaluate the physics form of the Hermite polynomial H_n(x)
    for a given integer n and array of real numbers x_array.

    Parameters
    ----------
    n : int
        Degree of the Hermite polynomial.
    x_array : array-like
        Real numbers for which to evaluate the polynomial.

    Returns
    -------
    np.array
        Evaluated polynomial values.
    """
    # Convert x_array to numpy array for vectorized operations
    x_array = np.array(x_array)
    
    # Base cases
    if n == 0:
        return np.ones_like(x_array)
    elif n == 1:
        return 2 * x_array

    # Recursive calculation
    h_prev = np.ones_like(x_array)
    h_curr = 2 * x_array
    for _ in range(2, n + 1):
        h_next = 2 * x_array * h_curr - 2 * (n - 1) * h_prev
        h_prev, h_curr = h_curr, h_next
    
    return h_curr