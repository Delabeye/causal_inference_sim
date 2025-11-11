def euclidean_distance_3d(p1, p2):
    """
    Calculate the 3-D Euclidean distance between two nodes
    :param p1: the first point (array-like with 3 elements)
    :param p2: the second point (array-like with 3 elements)
    :return: Euclidean distance between p1 and p2 (float)
    """

    dist = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5
    return dist

