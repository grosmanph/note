from utils.turb import Turb
import numpy as np

def generate_phase_screen(SR, H=1080, V=1080, w0=1, Pixel=8, n=0, nx=0, ny=0, seed=None):
    x = np.linspace(-H/2, H/2 -1, H)
    y = np.linspace(-V/2, V/2 -1, V)
    x = x * 8e-3
    y = y * 8e-3
    X, Y = np.meshgrid(x, y)
    phi = np.angle(X + 1j*Y)  # Azimuthal angle

    gx, gy = nx/H, ny/V
    turb = Turb(H, V, SR, w0, Pixel, rand_seed=seed)
    Hol = np.mod(turb + n * phi + 2 * np.pi * (Y * gy + X * gx), 2 * np.pi)
    Hol = Hol - Hol.min()
    Hol = Hol / np.max(Hol) * 255
    return turb, Hol
