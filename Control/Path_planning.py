import numpy as np
import random

class RRTNode:
    def __init__(self, pos, parent=None):
        self.pos = np.array(pos, dtype=float)
        self.parent = parent

class RRT3DPlanner:
    def __init__(
            self,
            config: dict,
        ):
        """
        bounds: dict { "x": (xmin, xmax), "y": (...), "z": (...) }
        step_size: distance step RRT per extension
        safe_distance : distance min par rapport à tout obstacle
        """
        self.bounds = config.get("world_bounds")
        self.step_size = config.get("step_size", 1.0)
        self.max_iter = config.get("max_iter", 3000)
        self.safe_distance = config.get("safe_distance", 1.0)

    # -------------------------------------------------------------
    # Utilitaires
    # -------------------------------------------------------------
    def _random_point(self):
        return np.array([
            random.uniform(*self.bounds["x"]),
            random.uniform(*self.bounds["y"]),
            random.uniform(*self.bounds["z"]),
        ])

    def _distance(self, a, b):
        return np.linalg.norm(a - b)

    def _nearest_node(self, nodes, point):
        dists = [self._distance(n.pos, point) for n in nodes]
        return nodes[np.argmin(dists)]

    def _steer(self, from_node, to_point):
        direction = to_point - from_node.pos
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return from_node.pos
        return from_node.pos + (direction / norm) * self.step_size

    def _is_collision_free(self, pos, obstacles):
        """Vérifie juste la distance aux obstacles (obstacles = liste de points)."""
        for ob in obstacles:
            if np.linalg.norm(pos - ob) < self.safe_distance:
                return False
        return True

    # -------------------------------------------------------------
    # Construire chemin final
    # -------------------------------------------------------------
    def _reconstruct_path(self, node):
        path = []
        cur = node
        while cur is not None:
            path.append(cur.pos)
            cur = cur.parent
        return path[::-1]

    # -------------------------------------------------------------
    # RRT principal
    # -------------------------------------------------------------
    def plan(self, start, goal, obstacles):
        start_node = RRTNode(start)
        nodes = [start_node]

        for _ in range(self.max_iter):

            # Sample random point
            rnd = self._random_point()

            # Get nearest RRT node
            nearest = self._nearest_node(nodes, rnd)

            # Move toward rnd
            new_pos = self._steer(nearest, rnd)

            # Check obstacle clearance
            if not self._is_collision_free(new_pos, obstacles):
                continue

            # Add new node
            new_node = RRTNode(new_pos, parent=nearest)
            nodes.append(new_node)

            # Check if goal reached
            if self._distance(new_node.pos, goal) < self.step_size:
                goal_node = RRTNode(goal, parent=new_node)
                return self._reconstruct_path(goal_node)

        return None  # Échec

