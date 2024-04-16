import numpy as np
from typing import List, Union
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import zoom

def Hermite(n):
    """
    Compute the Hermite polynomial H_n.

    Given a nonnegative integer n, compute the Hermite polynomial H_n. 
    The result is returned as a numpy array where the mth element is 
    the coefficient of x^(n+1-m). To evaluate H_n(x), use numpy's 
    polyval function.

    Parameters:
    -----------
    n : int
        Nonnegative integer representing the degree of Hermite polynomial.

    Returns:
    --------
    numpy.ndarray
        Coefficients of the Hermite polynomial of degree n.

    Example:
    --------
    >>> H4 = Hermite(4)
    >>> print(H4)
    [ 16.  -0.  12.  -0.   0.]

    >>> x = 2
    >>> value = np.polyval(H4, x)
    >>> print(value)
    60.0
    """

    # Base cases for n=0 and n=1
    if n == 0:
        return np.array([1])
    elif n == 1:
        return np.array([2, 0])

    # Initial setup for recurrence relation
    hkm2 = np.zeros(n+1)
    hkm2[-1] = 1
    hkm1 = np.zeros(n+1)
    hkm1[-2] = 2

    # Build the Hermite polynomial using recurrence relation
    for k in range(2, n+1):
        hk = np.zeros(n+1)
        
        # Calculate the polynomial coefficients
        for e in range(n-k, n+1, 2):
            hk[e] = 2 * (hkm1[e+1] - (k-1) * hkm2[e])

        hk[-1] = -2 * (k-1) * hkm2[-1]
        
        # Shift the polynomials for the next iteration
        if k < n:
            hkm2 = hkm1
            hkm1 = hk
                
    return hk


def Laguerre(p, l, x):
    """
    Generate Laguerre functions.

    This function computes the Laguerre functions based on given indices
    `p` and `l` and evaluates the resulting polynomial on a given vector `x`.

    Parameters:
    -----------
    p : int
        Index for the Laguerre function.
    l : int
        Index for the Laguerre function.
    x : numpy.ndarray
        Vector on which to evaluate the Laguerre function.

    Returns:
    --------
    numpy.ndarray
        Evaluated Laguerre function on vector `x`.

    Example:
    --------
    >>> result = Laguerre(2, 3, np.array([1, 2, 3]))
    >>> print(result)
    [values]
    """

    # Initialize y as zeros with p+1 elements
    y = np.zeros(p+1)

    # Check for base case where p is 0
    if p == 0:
        y[0] = 1
    else:
        # Loop over range of 0 to p to compute the coefficients
        for m in range(p+1):
            y[p - m] = ((-1)**m * np.math.factorial(p + l) /
                        (np.math.factorial(p - m) * np.math.factorial(l + m) * np.math.factorial(m)))

    # Evaluate the polynomial using coefficients in y at points x
    y_evaluated = np.polyval(y, x)
    return y_evaluated


def halve_element(var: Union[int, List], direction: str) -> Union[int, List]:
    """
    Halves an element in a nested or non-nested list based on the specified direction.

    Applies halving operation to the first element of each list or sublist
    if direction is 'horizontal', or to the second element if direction is 'vertical'.

    :param var: An integer or a (nested) list of integers to be partially halved.
    :type var: Union[int, List]
    :param direction: A string that determines which element to halve ('horizontal' or 'vertical').
    :type direction: str
    :return: A partially modified integer or (nested) list of integers, based on the direction.
    :rtype: Union[int, List]

    :Example:

    >>> halve_element([800, 600], 'horizontal')
    [400, 600]
    >>> halve_element([[1000, 800], [1200, 600]], 'horizontal')
    [[500, 800], [600, 600]]
    >>> halve_element([800, 600], 'vertical')
    [800, 300]
    >>> halve_element([[1000, 800], [1200, 600]], 'vertical')
    [[1000, 400], [1200, 300]]
    """

    if isinstance(var, list):
        # Apply the operation based on the direction to each sublist
        if direction == 'horizontal':
            return [[halve_element(subvar[0], direction) if isinstance(subvar, list) else subvar // 2] + subvar[1:] for subvar in var]
        elif direction == 'vertical':
            return [subvar[:1] + [halve_element(subvar[1], direction)] if len(subvar) > 1 else subvar for subvar in var]
    else:
        return var // 2


def fourier_rescale(image: np.ndarray, new_shape: tuple[int, int,]) -> np.ndarray:
    """
    Rescales an image using a Fourier-based method, preserving phase information.

    :param image: The input image as a 2D NumPy array.
    :type image: np.ndarray
    :param new_shape: The desired resolution of the output image as a tuple (width, height).
    :type new_shape: tuple[int, int]
    :return: The rescaled image as a 2D NumPy array.
    :rtype: np.ndarray

    The function performs the following steps:
    1. Computes the Fourier Transform of the input image.
    2. Applies a shift to center the zero-frequency component.
    3. Rescales the shifted Fourier image to the desired resolution.
    4. Applies an inverse shift and computes the Inverse Fourier Transform to get the rescaled image.
    """
    # Compute the Fourier Transform of the image
    fft_image = fft2(image)
    fft_image_shifted = fftshift(fft_image)

    # Calculate scaling factors for each dimension
    scale_x = new_shape[0] / image.shape[1]
    scale_y = new_shape[1] / image.shape[0]

    # Rescale the Fourier-transformed and shifted image
    rescaled_fft_image_shifted = zoom(fft_image_shifted, (scale_y, scale_x))

    # Compute the Inverse Fourier Transform after shifting back
    rescaled_fft_image = ifftshift(rescaled_fft_image_shifted)
    rescaled_image = ifft2(rescaled_fft_image)

    # Return the real part of the inverse transformed image
    return np.real(rescaled_image)


def upscale_image_nearest(image: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    """
    Upscale an image to a new resolution by nearest neighbor scaling.

    This function duplicates each pixel of the original image to fill a block of
    pixels in the upscaled image, maintaining the original content without
    interpolation.

    :param image: The original image as a 2D numpy array.
    :type image: np.ndarray
    :param new_width: The desired width of the upscaled image.
    :type new_width: int
    :param new_height: The desired height of the upscaled image.
    :type new_height: int
    :return: The upscaled image as a 2D numpy array.
    :rtype: np.ndarray

    The function calculates the best integer scale factors for both dimensions
    and uses these factors to duplicate the pixels of the original image.
    If the new dimensions are not exact multiples of the old dimensions,
    the excess from the upscaled image is trimmed.
    """
    # Calculate the scale factors for each dimension
    scale_x = new_width // image.shape[1]
    scale_y = new_height // image.shape[0]

    # Upscale the image by duplicating the rows and columns
    upscaled_image = np.repeat(image, scale_y, axis=0)
    upscaled_image = np.repeat(upscaled_image, scale_x, axis=1)

    # Trim the excess if the new dimensions are not exact multiples
    final_image = upscaled_image[:new_height, :new_width]

    return final_image

