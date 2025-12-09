import numpy as np
from pathfinding3d.core.grid import Grid
from pathfinding3d.finder.a_star import AStarFinder
from pathfinding3d.core.diagonal_movement import DiagonalMovement


class AStarPlanner:
    def __init__(self, config: dict, physics_client_id=0):
        self.physics_client_id = physics_client_id 
        self.bounds = config.get("world_bounds")
        if not self.bounds: raise ValueError("Config A* doit inclure 'world_bounds'")

        self.resolution = 0.25
        # [MODIF] Réduction de la marge pour éviter les blocages excessifs
        self.safety_margin_cells = 1 

        self.min_x, self.max_x = self.bounds["x"]
        self.min_y, self.max_y = self.bounds["y"]
        self.min_z, self.max_z = self.bounds["z"]
        
        self.width = int(np.ceil((self.max_x - self.min_x) / self.resolution))
        self.height = int(np.ceil((self.max_y - self.min_y) / self.resolution))
        self.depth = int(np.ceil((self.max_z - self.min_z) / self.resolution))
        
        self.finder = AStarFinder(diagonal_movement=DiagonalMovement.always)

    def _pos_to_node(self, pos):
        x = int((pos[0] - self.min_x) / self.resolution)
        y = int((pos[1] - self.min_y) / self.resolution)
        z = int((pos[2] - self.min_z) / self.resolution)
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        z = max(0, min(z, self.depth - 1))
        return x, y, z

    def _node_to_pos(self, node):
        px = (node.x * self.resolution) + self.min_x
        py = (node.y * self.resolution) + self.min_y
        pz = (node.z * self.resolution) + self.min_z
        return np.array([px, py, pz])

    def _build_grid_from_cloud(self, point_cloud):
        matrix = np.ones((self.width, self.height, self.depth), dtype=np.int8)
        for p in point_cloud:
            cx, cy, cz = self._pos_to_node(p)
            x0 = max(0, cx - self.safety_margin_cells)
            x1 = min(self.width, cx + self.safety_margin_cells + 1)
            y0 = max(0, cy - self.safety_margin_cells)
            y1 = min(self.height, cy + self.safety_margin_cells + 1)
            z0 = max(0, cz - self.safety_margin_cells)
            z1 = min(self.depth, cz + self.safety_margin_cells + 1)
            matrix[x0:x1, y0:y1, z0:z1] = 0 
        return Grid(matrix=matrix)

    def _is_line_safe_on_grid(self, grid, start_pos, end_pos):
        dist = np.linalg.norm(end_pos - start_pos)
        if dist < 0.05: return True
        steps = int(np.ceil(dist / (self.resolution / 2)))
        for i in range(steps + 1):
            t = i / steps
            pt = start_pos + (end_pos - start_pos) * t
            nx, ny, nz = self._pos_to_node(pt)
            if not grid.node(nx, ny, nz).walkable:
                return False
        return True

    def check_line_validity(self, start_pos, end_pos, map_points):
        grid = self._build_grid_from_cloud(map_points)
        return self._is_line_safe_on_grid(grid, np.array(start_pos), np.array(end_pos))

    def check_path_validity(self, path, map_points):
        grid = self._build_grid_from_cloud(map_points)
        for i in range(len(path)-1):
            if not self._is_line_safe_on_grid(grid, np.array(path[i]), np.array(path[i+1])):
                return False
        return True

    def _prune_path(self, path, grid):
        if len(path) < 3: return path
        pruned = [path[0]]
        curr = 0
        while curr < len(path) - 1:
            for i in range(len(path) - 1, curr, -1):
                if self._is_line_safe_on_grid(grid, path[curr], path[i]):
                    pruned.append(path[i]); curr = i; break
            else: curr += 1; pruned.append(path[curr])
        return pruned

    def _resample_path(self, path, spacing=0.4):
        if len(path) < 2: return path
        new_path = [path[0]]
        for i in range(len(path) - 1):
            p0 = np.array(path[i]); p1 = np.array(path[i+1])
            dist = np.linalg.norm(p1 - p0)
            if dist > spacing:
                num = int(np.ceil(dist / spacing))
                for j in range(1, num + 1): new_path.append(p0 + (p1 - p0) * (j / num))
            else: new_path.append(p1)
        return new_path

    def plan(self, start, goal, map_points=[], smooth=True):
        start = np.array(start); goal = np.array(goal)
        grid = self._build_grid_from_cloud(map_points)
        
        # Check direct
        if self._is_line_safe_on_grid(grid, start, goal):
            return self._resample_path([start, goal], spacing=0.4)

        sx, sy, sz = self._pos_to_node(start)
        gx, gy, gz = self._pos_to_node(goal)
        start_node = grid.node(sx, sy, sz)
        end_node = grid.node(gx, gy, gz)
        
        if not start_node.walkable:
            for n in grid.neighbors(start_node):
                if n.walkable: start_node = n; break
            else: return None
        if not end_node.walkable: return None

        path_nodes, _ = self.finder.find_path(start_node, end_node, grid)
        if not path_nodes: return None
            
        path = [self._node_to_pos(n) for n in path_nodes]
        if np.linalg.norm(path[0] - start) > 0.1: path.insert(0, start)
        if np.linalg.norm(path[-1] - goal) > 0.1: path.append(goal)

        if smooth:
            pruned = self._prune_path(path, grid)
            print (pruned)
            return self._resample_path(pruned, spacing=0.4)
        print (path)
        return self._resample_path(path)