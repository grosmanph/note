import numpy as np
from scipy.special import j1
from scipy.optimize import newton, brentq

def inverse_bessel_j1(y: float, x0: float = 0.9, method: str = 'newton') -> float:
    """
    Calculate the inverse of the first-order Bessel function J1.

    Parameters
    ----------
    y : float
        The value for which the inverse of the Bessel function is to be computed.
    x0 : float, optional
        The initial guess for the root-finding algorithm (default is 1.0).
    method : str, optional
        The root-finding method to use ('newton' or 'brentq').

    Returns
    -------
    float
        The value x such that J1(x) is approximately equal to y, or NaN if
        the root-finding algorithm fails to converge.

    """
    def func(x: float) -> float:
        return j1(x) - y

    try:
        if method == 'newton':
            return newton(func, x0, maxiter=100)
        elif method == 'brentq':
            # Define a range for the root. Adjust these values as needed.
            return brentq(func, a=-5, b=5)
        else:
            raise ValueError("Invalid method specified.")
    except RuntimeError:
        return np.nan  # or handle the error as needed


def apply_inverse_bessel_j1_2d(array: np.ndarray) -> np.ndarray:
    """
    Apply the inverse Bessel function J1 to each element of a 2D array.

    Parameters
    ----------
    array : np.ndarray
        A 2D array containing the values for which the inverse of J1 is required.

    Returns
    -------
    np.ndarray
        A 2D array with the inverse Bessel function applied to each element.

    """
    # Determine the shape of the input array
    shape = array.shape
    result = np.empty(shape)

    # Apply the inverse Bessel function to each element
    for i in range(shape[0]):
        for j in range(shape[1]):
            if array[i, j] < 0.001:
                result[i, j] = 0
            else:
                result[i, j] = inverse_bessel_j1(array[i, j])

    return result
