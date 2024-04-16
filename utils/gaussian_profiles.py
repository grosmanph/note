import numpy as np
import cmath, math
from utils.polynomials import generalized_laguerre, hermite_physics

def w(z: float, w0: float, k: float, diverging: bool = True) -> float:
    """
    Evolving beam width of a Gaussian beam.

    Parameters:
    z : float
        Propagation distance (mm).
    w0 : float
        Beam waist at z=0 (mm).
    k : float
        Wavenumber (mm^-1).

    Returns:
    float
        Beam width at distance z.
    """
    if diverging:
        lambda_ = 2 * np.pi / k
        return w0 * np.sqrt(1 + (z * lambda_ / (np.pi * w0**2))**2)
    else:
        return w0


def R(z: float, w0: float, k: float, diverging: bool = True) -> float:
    """
    Radius of curvature of a Gaussian beam.

    Parameters:
    z : float
        Propagation distance (mm).
    w0 : float
        Beam waist at z=0 (mm).
    k : float
        Wavenumber (mm^-1).

    Returns:
    float
        Radius of curvature at distance z.
    """
    if diverging:
        lambda_ = 2 * np.pi / k
        return z * (1 + (np.pi * w0**2 / (z * lambda_))**2)
    else:
        return 10000000


def gouy_phase(z: float, w0: float, k: float) -> float:
    """
    Gouy phase shift of a Gaussian beam.

    Parameters:
    z : float
        Propagation distance (mm).
    w0 : float
        Beam waist at z=0 (mm).
    k : float
        Wavenumber (mm^-1).

    Returns:
    float
        Gouy phase shift at distance z.
    """
    lambda_ = 2 * np.pi / k
    return np.arctan(z * lambda_ / (np.pi * w0**2))

def HG_transverse_complex_amplitude(
    x: np.ndarray,
    y: np.ndarray,
    z: float,
    k: float,
    w0: float,
    m: int,
    n: int,
    diverging: bool = True
) -> np.ndarray:
    """
    Calculate the transverse complex amplitude of Hermite-Gaussian beams.

    Parameters
    ----------
    x : np.ndarray
        1D array representing the x spatial coordinates in millimeters (mm).
    y : np.ndarray
        1D array representing the y spatial coordinates in millimeters (mm).
    z : float
        Propagation distance along the z-direction in millimeters (mm).
    k : float
        Wavenumber of the beam in inverse millimeters (mm^-1).
    w0 : float
        Initial waist (width) of the beam at z = 0 in millimeters (mm).
    m : int
        Non-negative integer representing the order of the Hermite polynomial in x-direction (dimensionless).
    n : int
        Non-negative integer representing the order of the Hermite polynomial in y-direction (dimensionless).

    Returns
    -------
    np.ndarray
        2D array representing the unitless complex amplitude of Hermite-Gaussian beams at the specified spatial coordinates.

    Notes
    -----
    The output complex amplitude array is calculated based on the evolving beam width `w(z)`, 
    the radius of curvature `R(z)`, and the Gouy phase `gouy_phase(z)`, which should be defined externally.
    Ensure all input parameters are provided in consistent units to achieve a unitless return output.
    """

    # Get the beam width, radius of curvature, and Gouy phase
    w_z = w(z, w0, k, diverging)
    R_z = R(z, w0, k, diverging)
    gouy = gouy_phase(z, w0, k)
    
    # Calculate waist ratio
    waist_ratio = w0 / w_z

    # Calculate the Hermite polynomial terms
    H_m = hermite_physics(m, np.sqrt(2)*x/w_z)
    H_n = hermite_physics(n, np.sqrt(2)*y/w_z)

    # Gaussian envelope
    gaussian_term = np.exp(-1 * (x**2 + y**2) / w_z**2)

    # Complex curvature term
    complex_term = np.exp(-1j * k * (x**2 + y**2) / (2 * R_z))
    
    # Phase terms
    phase_term = cmath.exp(1j * gouy)
    plane_wave_term = cmath.exp(-1j * k * z)

    # Compute the total complex amplitude
    E_complex = waist_ratio * H_m * H_n * gaussian_term
    E_complex = E_complex.astype(np.complex128)  # Ensure it's complex
    E_complex *= complex_term * phase_term * plane_wave_term
    
    return E_complex


def LG_transverse_complex_amplitude(
    r: np.ndarray,
    theta: np.ndarray,
    z: float,
    k: float,
    w0: float,
    p: int,
    l: int,
    diverging: bool = True
) -> np.ndarray:
    """
    Calculate the transverse complex amplitude of Laguerre-Gaussian beams.

    Parameters
    ----------
    r : np.ndarray
        2D array representing the radial coordinates in millimeters (mm).
    theta : np.ndarray
        2D array representing the azimuthal coordinates in radians (rad).
    z : float
        Propagation distance along the z-direction in millimeters (mm).
    k : float
        Wavenumber of the beam in inverse millimeters (mm^-1).
    w0 : float
        Initial waist (width) of the beam at z = 0 in millimeters (mm).
    p : int
        Non-negative integer representing the radial index.
    l : int
        Integer representing the azimuthal index.

    Returns
    -------
    np.ndarray
        2D array representing the unitless complex amplitude of Laguerre-Gaussian beams at the specified spatial coordinates.

    Notes
    -----
    The output complex amplitude array is calculated based on the evolving beam width `w(z)`, 
    the radius of curvature `R(z)`, and the Gouy phase `gouy_phase(z)`, which should be defined externally.
    Ensure all input parameters are provided in consistent units to achieve a unitless return output.
    """

    # Get the beam width, radius of curvature, and Gouy phase
    w_z = w(z, w0, k, diverging)
    R_z = R(z, w0, k, diverging)
    gouy = gouy_phase(z, w0, k)
    
    # Calculate waist ratio and other factors
    waist_ratio = w0 / w_z
    normalization_factor = np.sqrt(2 * math.factorial(p) / (np.pi * math.factorial(p + abs(l))))
    radial_factor = (np.sqrt(2)*r/w_z)**abs(l)

    # Calculate the Laguerre polynomial terms
    L_pl = generalized_laguerre(abs(l), p, 2*r**2/w_z**2)

    # Gaussian envelope
    gaussian_term = np.exp(-1 * r**2 / w_z**2)

    # Azimuthal phase
    azimuthal_term = np.exp(-1j * l * theta)

    # Complex curvature term
    complex_term = np.exp(-1j * k * r**2 * z / (2 * R_z))

    # Phase terms
    phase_term = cmath.exp(1j * gouy)

    # Compute the total complex amplitude
    E_complex = normalization_factor * waist_ratio * radial_factor * gaussian_term * L_pl
    E_complex = E_complex.astype(np.complex128)  # Ensure it's complex
    E_complex *= complex_term * azimuthal_term * phase_term
    
    return E_complex