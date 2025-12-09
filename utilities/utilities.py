import numpy as np

def euclidean_distance_3d(p1, p2):
    """
    Calculate the 3-D Euclidean distance between two nodes
    :param p1: the first point (array-like with 3 elements)
    :param p2: the second point (array-like with 3 elements)
    :return: Euclidean distance between p1 and p2 (float)
    """

    dist = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5
    return dist

def point_in_cube(point, cube):
    """
    Check if a point is inside a cube defined by its center and size.
    :param point: The point to check (array-like with 3 elements)
    :param cube: A dictionary with 'center' (array-like with 3 elements) and 'size' (float)
    :return: True if the point is inside the cube, False otherwise
    """
    center = np.array(cube['center'])
    size = np.array([cube['length'], cube['width'], cube['height']]) / 2.0  # Half-size for easier calculations
    point = np.array(point)

    if all(center - size <= point) and all(point <= center + size):
        return True
    return False

def point_in_cylinder(point, cylinder):
    """
    Check if a point is inside a cylinder defined by its center, radius, and height.
    :param point: The point to check (array-like with 3 elements)
    :param cylinder: A dictionary with 'center' (array-like with 3 elements), 'radius' (float), and 'height' (float)
    :return: True if the point is inside the cylinder, False otherwise
    """
    center = np.array(cylinder['center'])
    radius = cylinder['radius']
    height = cylinder['height']
    point = np.array(point)

    # Check horizontal distance from center
    horizontal_dist = np.linalg.norm(point[:2] - center[:2])
    vertical_dist = abs(point[2] - center[2])

    if horizontal_dist <= radius and vertical_dist <= (height / 2.0):
        return True
    return False