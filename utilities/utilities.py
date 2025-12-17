import numpy as np

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

def discretize_obstacles(obstacles):
        points = []
        res = 0.25
        for obs in obstacles:
            center = np.array(obs["center"])
            otype = obs["type"]
            # Discrétisation simplifiée
            if otype == "cube":
                l, w, h = obs["length"], obs["width"], obs["height"]
                xs = np.arange(center[0]-l/2, center[0]+l/2+res, res)
                ys = np.arange(center[1]-w/2, center[1]+w/2+res, res)
                zs = np.arange(center[2]-h/2, center[2]+h/2+res, res)
                for x in xs:
                    for y in ys:
                        for z in zs: points.append([round(x, 2), round(y, 2), round(z, 2)])
            elif otype == "sphere":
                r = obs["radius"]
                # Approximation cubique pour aller vite au démarrage
                xs = np.arange(center[0]-r, center[0]+r+res, res)
                ys = np.arange(center[1]-r, center[1]+r+res, res)
                zs = np.arange(center[2]-r, center[2]+r+res, res)
                for x in xs:
                    for y in ys:
                        for z in zs:
                            if np.linalg.norm(np.array([x,y,z])-center) <= r: points.append([round(x, 2), round(y, 2), round(z, 2)])
            elif otype == "cylinder":
                r, h = obs["radius"], obs["height"]
                xs = np.arange(center[0]-r, center[0]+r+res, res)
                ys = np.arange(center[1]-r, center[1]+r+res, res)
                zs = np.arange(center[2]-h/2, center[2]+h/2+res, res)
                for x in xs:
                    for y in ys:
                        if np.linalg.norm(np.array([x,y])-center[:2]) <= r:
                            for z in zs: points.append([round(x, 2), round(y, 2), round(z, 2)])
        return points