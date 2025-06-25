from math import atan2, degrees, ceil
import matplotlib
import matplotlib.pyplot as plt
import psutil
from scipy.spatial import cKDTree
import numpy as np
from .CartImage import CartImage
import os


def solve_scaling_magnitude(camera_scale, rot_angle_deg):
    """
    Computes the scaling magnitude based on the camera scale and rotation angle.
    The function solves the equations:
        x + y = camera_scale
        sqrt(x^2 + y^2) = y / sin(theta)

    Returns:
        float: The value of sqrt(x^2 + y^2), using the solution with x > 0 and y > 0.
    """
    theta = np.radians(rot_angle_deg)
    s = camera_scale

    A = 2 * np.sin(theta)**2 - 1
    B = -2 * s * np.sin(theta)**2
    C = s**2 * np.sin(theta)**2

    discriminant = B**2 - 4 * A * C

    if discriminant < 0:
        raise ValueError("No real solution exists")

    sqrt_discriminant = np.sqrt(discriminant)

    # Try both roots
    y1 = (-B + sqrt_discriminant) / (2 * A)
    y2 = (-B - sqrt_discriminant) / (2 * A)

    candidates = []
    for y in (y1, y2):
        x = s - y
        if x > 0 and y > 0:
            candidates.append((x, y))

    if not candidates:
        raise ValueError("No positive solution for both x and y")

    # Use the first valid (x, y) pair
    x, y = candidates[0]
    return float(np.sqrt(x**2 + y**2))

