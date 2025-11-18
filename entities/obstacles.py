class SphericalObstacle:
    
    """Class representing a spherical obstacle in the environment."""

    def __init__(self, center, radius, obstacle_id=1):
        self.center = center  # in meter
        self.radius = radius  # in meter
        self.id = obstacle_id


class CubeObstacle:
    
    """Class representing a cubic obstacle in the environment."""
    
    def __init__(self, center, length, width, height, obstacle_id=2):
        self.center = center
        self.length = length
        self.width = width
        self.height = height
        self.id = obstacle_id


class CylindricalObstacle:

    """Class representing a cylindrical obstacle in the environment."""
    
    def __init__(self, center, radius, height, obstacle_id=3):
        self.center = center
        self.radius = radius
        self.height = height
        self.id = obstacle_id