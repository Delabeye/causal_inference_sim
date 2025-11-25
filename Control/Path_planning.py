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
        bounds: dict { "x": (xmin, xmax), "y": (...), "z": (...) }  OR key "world_bounds"
        step_size: distance step RRT per extension
        safe_distance : distance min par rapport à tout obstacle
        """
        # accept either "world_bounds" or "bounds"
        self.bounds = config.get("world_bounds") or config.get("bounds")
        if not self.bounds:
            raise ValueError("Config must include 'world_bounds' or 'bounds' as dict with keys 'x','y','z'.")

        self.step_size = config.get("step_size", 1.0)
        self.max_iter = config.get("max_iter", 3000)
        self.safe_distance = config.get("safe_distance", 1.0)
        # probability to sample the goal directly (helps to find path)
        self.goal_sample_rate = config.get("goal_sample_rate", 0.05)

    # -------------------------------------------------------------
    # Utilitaires
    # -------------------------------------------------------------
    def _random_point(self):
        return np.array([
            random.uniform(*self.bounds["x"]),
            random.uniform(*self.bounds["y"]),
            random.uniform(*self.bounds["z"]),
        ])

    def _clip_to_bounds(self, p):
        return np.array([
            np.clip(p[0], *self.bounds["x"]),
            np.clip(p[1], *self.bounds["y"]),
            np.clip(p[2], *self.bounds["z"]),
        ])

    def _distance(self, a, b):
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        return np.linalg.norm(a - b)

    def _nearest_node(self, nodes, point):
        dists = [self._distance(n.pos, point) for n in nodes]
        return nodes[int(np.argmin(dists))]

    def _steer(self, from_node, to_point):
        from_pos = from_node.pos
        to_point = np.array(to_point, dtype=float)
        direction = to_point - from_pos
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return from_pos.copy()
        step = min(self.step_size, norm)
        new_pos = from_pos + (direction / norm) * step
        return self._clip_to_bounds(new_pos)

    def _is_collision_free(self, pos, obstacles):
        """Vérifie juste la distance aux obstacles (obstacles = liste de points)."""
        for ob in obstacles:
            if np.linalg.norm(pos - ob) < self.safe_distance:
                return False
        return True

    def _is_path_collision_free(self, a, b, obstacles):
        """Check collisions along segment [a,b] by sampling points."""
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        dist = self._distance(a, b)
        if dist < 1e-8:
            return self._is_collision_free(a, obstacles)
        # sample at resolution half of safe_distance (or a few points)
        step = max(self.safe_distance * 0.5, 0.1)
        num = int(np.ceil(dist / step))
        for i in range(num + 1):
            t = i / max(num, 1)
            p = a + t * (b - a)
            if not self._is_collision_free(p, obstacles):
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
        # normalize inputs
        start_np = np.array(start, dtype=float)
        goal_np = np.array(goal, dtype=float)
        obs_np = [np.array(o, dtype=float) for o in (obstacles or [])]

        # quick checks
        if not self._is_collision_free(start_np, obs_np):
            return None
        if not self._is_collision_free(goal_np, obs_np):
            return None

        start_node = RRTNode(start_np)
        nodes = [start_node]

        for _ in range(self.max_iter):

            # Sample random point (with small goal bias)
            if random.random() < self.goal_sample_rate:
                rnd = goal_np
            else:
                rnd = self._random_point()

            # Get nearest RRT node
            nearest = self._nearest_node(nodes, rnd)

            # Move toward rnd
            new_pos = self._steer(nearest, rnd)

            # Check obstacle clearance along the segment from nearest to new_pos
            if not self._is_path_collision_free(nearest.pos, new_pos, obs_np):
                continue

            # Add new node
            new_node = RRTNode(new_pos, parent=nearest)
            nodes.append(new_node)

            # Check if goal reached (and path from new_node to goal is collision-free)
            if self._distance(new_node.pos, goal_np) <= self.step_size:
                if self._is_path_collision_free(new_node.pos, goal_np, obs_np):
                    goal_node = RRTNode(goal_np, parent=new_node)
                    return self._reconstruct_path(goal_node)

        return None  # Échec

