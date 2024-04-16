import numpy as np

import numpy as np

def Turb(H, V, SR, w0, Pixel, rand_seed=None):
    # H: Number of pixels in the horizontal direction
    # V: Number of pixels in the vertical direction
    # SR: Strehl ratio with values in the inverval [0,1]
    # w0: Beam size at the SLM in mm
    # Pixel size in micrometers
    if rand_seed is not None:
        np.random.seed(rand_seed)  # Set the random seed if rand_seed is specified
    Size = min(H, V)
    w0 = (w0 / 1000) / (Pixel / 1000000)  # Gaussian beam radius (in pixels)
    r0 = w0 / (((1 / SR) - 1) / 6.88) ** (3/5)  # Fried's Parameter

    # Number of points for square area
    Delta = 1 / (Pixel * Size)  # increment size for x and y

    # put zero (origin) between samples to avoid singularity
    nx, ny = np.meshgrid(np.arange(1, Size+1) - Size/2 - 0.5, np.arange(1, Size+1) - Size/2 - 0.5)
    Modgrid = np.real(np.exp(-1j * np.pi * (nx + ny)))
    rr = (nx ** 2 + ny ** 2) * Delta ** 2

    # Square root of the Kolmogorov spectrum:
    qKol = 0.1516 * Delta / r0 ** (5/6) * rr ** (-11/12)

    # The first f0 definition, based on the random seed (if provided)
    f0 = (np.random.randn(Size, Size) + 1j * np.random.randn(Size, Size)) * qKol / np.sqrt(2)
    f1 = np.fft.fft2(f0) * Modgrid

    ary = np.array([-0.25, -0.25, -0.25, -0.125, -0.125, -0.125, 0, 0, 0, 0, 0.125, 0.125, 0.125, 0.25, 0.25, 0.25])
    bry = np.array([-0.25, 0, 0.25, -0.125, 0, 0.125, -0.25, -0.125, 0.125, 0.25, -0.125, 0, 0.125, -0.25, 0, 0.25])
    dary = np.full(16, 0.25)
    dbry = np.full(16, 0.25)

    ss = (ary**2 + bry**2) * Delta**2
    qsKol = 0.1516 * Delta / r0 ** (5/6) * ss ** (-11/12)

    # The second f0 definition, affected by the random seed (if provided)
    f0 = (np.random.randn(16) + 1j * np.random.randn(16)) * qsKol / np.sqrt(2)
    fn = f1  # numpy.zeros((Size, Size))
    for pp in range(16):
        eks = np.exp(1j * 2 * np.pi * (nx * ary[pp] + ny * bry[pp]) / Size)
        fn += f0[pp] * eks * dary[pp] * dbry[pp]

    ff = np.zeros((Size, Size))
    ff[int(Size/2 - Size/2):int(Size/2 + Size/2), int(Size/2 - Size/2):int(Size/2 + Size/2)] = np.real(fn)
    yo = (H-Size)//2
    yf = yo + Size
    xo = (V-Size)//2
    xf = xo + Size
    T = np.zeros((V, H))
    turb0 = ff[:, :Size]
    T[xo:xf, yo:yf] = turb0
    turb = T
    return turb


