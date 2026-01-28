import numpy as np
from scipy.spatial import KDTree
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import pybullet as p

class HeightmapAStar:
    """
    A hybrid navigation and path-planning class for autonomous UAVs.

    This class integrates a 2.5D heightmap for global pathfinding (A*) with local 
    Artificial Potential Fields (APF) to handle dynamic obstacle avoidance and 
    swarm separation. It manages the conversion between physical world coordinates 
    and discrete grid indices.

    Attributes:
        res (float): Grid resolution in meters per cell.
        bounds (dict): World boundaries for X, Y, and Z axes.
        width (int): Grid dimension along the X-axis.
        height (int): Grid dimension along the Y-axis.
        h_map (np.ndarray): 2D float32 array representing the maximum altitude at each cell.
        safety_margin (float): Geometric inflation radius (meters) for obstacles.
    """
    def __init__(self, config, resolution=0.25):
        """
        Initializes the navigation engine with spatial constraints.

        Args:
            config (dict): Configuration dictionary containing 'world_bounds' and 'safety_margin'.
            resolution (float, optional): Spatial resolution of the grid in meters. Defaults to 0.25.
        """
        self.res = resolution
        self.bounds = config.get("world_bounds", {'x': [-500, 500], 'y': [-500, 500], 'z': [0, 100]})
        self.min_x, self.min_y = self.bounds['x'][0], self.bounds['y'][0]
        
        # Grid dimensions
        self.width = int((self.bounds['x'][1] - self.min_x) / self.res)
        self.height = int((self.bounds['y'][1] - self.min_y) / self.res)
        
        # Heightmap en float32
        self.h_map = np.zeros((self.width, self.height), dtype=np.float32)
        
        self.finder = AStarFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle)
        
        # Recovery of margin in METRES (e.g. 1.0m)
        self.safety_margin = config.get("safety_margin", 1.0)

    def build_from_buildings(self, buildings):
        """
        Constructs the heightmap using geometric inflation of building primitives.

        Increases the footprint of each building by the safety margin before mapping 
        it to the grid to ensure a collision-free buffer for the UAV.

        Args:
            buildings (dict|list): Collection of building data containing 'center', 
                           'height', 'width', and 'length'.
        """
        source_data = buildings.values() if isinstance(buildings, dict) else buildings
        centers = []
        
        for data in source_data:
            center = data["center"]
            centers.append(center[:2])
            
            # Real Dimensions
            h = data["height"]
            real_w = data["width"]
            real_l = data["length"]
            
            # --- GEOMETRIC INFLATION
            # A safety margin is added on both sides.
            # If the margin is 1 metre, the building becomes 2 metres wider and longer.
            effective_w = real_w + (2.0 * self.safety_margin)
            effective_l = real_l + (2.0 * self.safety_margin)
            
            # Calculation of indices with inflated dimensions
            # X-axis (Width)
            x_min = int((center[0] - effective_w/2 - self.min_x) / self.res)
            x_max = int((center[0] + effective_w/2 - self.min_x) / self.res)
            
            # Y-axis (Length)
            y_min = int((center[1] - effective_l/2 - self.min_y) / self.res)
            y_max = int((center[1] + effective_l/2 - self.min_y) / self.res)
            
            # Clamping to stay on the map
            x0 = max(0, x_min); x1 = min(self.width, x_max)
            y0 = max(0, y_min); y1 = min(self.height, y_max)
            
            # Filling
            if x0 < x1 and y0 < y1:
                self.h_map[x0:x1, y0:y1] = np.maximum(self.h_map[x0:x1, y0:y1], h)
                
        if centers:
            self.building_tree = KDTree(np.array(centers))
    
    def custom_heightmap (self):
        """
        Generates a high-fidelity heightmap using PyBullet ray-casting.

        Scans the environment by firing batches of vertical rays to detect the 
        ground truth altitude of the terrain and all loaded URDF/SDF objects.
        """
        z_start = self.bounds['z'][1] + 100
        z_end = self.bounds['z'][0] - 1.0 
    
        # We scan the grid line by line (X) to send packets of rays (Y).
        for i in range(self.width):
            ray_starts = []
            ray_ends = []
        
            #  Calculation of the current X world
            curr_x = self.min_x + i * self.res
        
            for j in range(self.height):
            #  Calculation of the current Y world
                curr_y = self.min_y + j * self.res
            
                ray_starts.append([curr_x, curr_y, z_start])
                ray_ends.append([curr_x, curr_y, z_end])
            
        # Send to PyBullet
            results = p.rayTestBatch(ray_starts, ray_ends)
        
        # Get altitude for each ray
            for j, res in enumerate(results):
                hit_fraction = res[2]
                hit_pos = res[3]
            
                if hit_fraction < 1.0:
                # We store the Z altitude of the impact.
                    self.h_map[i, j] = hit_pos[2]
                else:
                # If nothing is touched, the ground is considered (Z=0).
                    self.h_map[i, j] = 0.0
                
            if i % 400 == 0: 
                print(f"Progression : {int(i / self.width * 100)}%")

        print("Heightmap succesfully generated.")
        

    def compute_repulsive_force(self, current_pos, safety_radius, max_force, swarm_active, leader, swarm_pos):
        """
        Calculates the local steering vector based on repulsive potential fields.

        Evaluates the local neighborhood of the agent to generate three types of forces:
        1. Lateral Repulsion: Pushes away from walls or high-altitude obstacles.
        2. Vertical Lift: Prevents collisions with the ground or building rooftops.
        3. Swarm Separation: Maintains distance between agents in a multi-UAV setup.

        Args:
            current_pos (np.ndarray): Current UAV position $[x, y, z]$ in meters.
            safety_radius (float): The influence radius of the potential field.
            max_force (float): Maximum allowable magnitude for the resulting vector.
            swarm_active (bool): Enables inter-agent collision avoidance.
            leader (bool): If True, the agent ignores swarm repulsion from followers.
            swarm_pos (dict): Map of neighbor IDs to their respective $[x, y, z]$ positions.

        Returns:
            np.ndarray: A 3D force vector $[f_x, f_y, f_z]$ normalized to max_force.
        """
        force_vec = np.array([0.0, 0.0, 0.0])
        k_obs = 0.5
        rows, cols = self.h_map.shape
    
        # Conversion from world position to grid index (with (0,0) at the centre)
        # We use // to obtain an integer (floor division)
        ix = int(current_pos[0] / self.res + rows / 2)
        iy = int(current_pos[1] / self.res + cols / 2)
        window_px = int(safety_radius / self.res)
    
        # Window boundaries (secured so as not to exit the matrix)
        x_min, x_max = max(0, ix - window_px), min(rows, ix + window_px + 1)
        y_min, y_max = max(0, iy - window_px), min(cols, iy + window_px + 1)
    
        # Extraction of the local area
        local_h_map = self.h_map[x_min:x_max, y_min:y_max]
    
        # Calculate the actual coordinates of each pixel in the extracted area
        # (We perform the reverse operation to find the metre from the index)
        x_range = (np.arange(x_min, x_max) - rows / 2) * self.res
        y_range = (np.arange(y_min, y_max) - cols / 2) * self.res
        X, Y = np.meshgrid(x_range, y_range, indexing='ij')
        
        DX = current_pos[0] - X
        DY = current_pos[1] - Y
        Dist_horizontale = np.sqrt(DX**2 + DY**2)

        # --- DECISION LOGIC ---
        
        # Basic filter: points within the safety radius
        mask_near = (Dist_horizontale < safety_radius) & (Dist_horizontale > 0.1)

        # CASE A: The point is ABOVE the drone (wall/high obstacle) -> Push to the side (XY)
        mask_wall = mask_near & (local_h_map >= current_pos[2])
        n_wall_pts = np.sum(mask_wall)
        
        if n_wall_pts > 0:
            # We calculate the average force so as not to blow up the meters.
            # Use of quadratic decay for greater smoothness.
            mags = (1.0 - (Dist_horizontale[mask_wall] / safety_radius))**2
            weights = mags / (Dist_horizontale[mask_wall] + 0.01)
            sum_weights = np.sum(weights)
            
            # Pure repulsion vector (pushes backwards)
            fx_rep = np.sum((DX[mask_wall] / Dist_horizontale[mask_wall]) * weights * max_force) / sum_weights
            fy_rep = np.sum((DY[mask_wall] / Dist_horizontale[mask_wall]) * weights * max_force) / sum_weights
            
            k_glide = 0.5  # Adjust between 0.2 and 0.8
            fx_glide = -fy_rep * k_glide
            fy_glide =  fx_rep * k_glide
            
            # 3. Applied combined force
            force_vec[0] += (fx_rep + fx_glide) * k_obs
            force_vec[1] += (fy_rep + fy_glide) * k_obs

        # CASE B: The point is BELOW the drone (Ground/Roof) -> Push upwards (Z)
        # Only considered if vertically close (e.g. margin of 2.0m)
        v_margin = 2.5           # Vertical zone of influence (metres)
        ground_threshold = 1   # Below 2m, we consider it to be the ground
        
        # Basic filter: points under the drone and within the influence zone
        mask_below = mask_near & (local_h_map < current_pos[2]) & (local_h_map > current_pos[2] - v_margin)

        if np.any(mask_below):
            # CASE B1: It is the GROUND (low altitude)
            mask_is_ground = mask_below & (local_h_map < ground_threshold)
            
            # CASE B2: It is a BUILDING/OBSTACLE (High altitude but below the drone)
            mask_is_roof = mask_below & (local_h_map >= ground_threshold)

            # Behaviour for ground: Gentle force for maintaining altitude
            if np.any(mask_is_ground):
                dist_v_ground = current_pos[2] - local_h_map[mask_is_ground]
                mag_ground = (1.0 - (dist_v_ground / v_margin))**2
                # A lower gain (k_ground) is applied to prevent the drone from ‘jumping’.
                force_vec[2] += np.mean(mag_ground * max_force) * 0.3 

            # Behaviour for a ROOF: Stronger force to avoid collision
            if np.any(mask_is_roof):
                dist_v_roof = current_pos[2] - local_h_map[mask_is_roof]
                mag_roof = (1.0 - (dist_v_roof / v_margin))**2
                # A greater force (k_roof) is applied because the impact is more dangerous.
                force_vec[2] += np.mean(mag_roof * max_force) * 1.2
            
                # horizontal force (XY) to make the drone move away from the edges of the roof
                force_vec[0] += np.mean((DX[mask_is_roof] / Dist_horizontale[mask_is_roof]) * mag_roof * max_force) * 0.2
                force_vec[1] += np.mean((DY[mask_is_roof] / Dist_horizontale[mask_is_roof]) * mag_roof * max_force) * 0.2
        if swarm_active and not leader:
            for _,other_pos in swarm_pos.items():
                diff = current_pos - other_pos
                dist_uav = np.linalg.norm(diff)
                if dist_uav < safety_radius:
                    mag = (1.0 - (dist_uav / safety_radius))
                    force_vec += ((diff / dist_uav) * mag * max_force)/2

        # Final normalization
        total_norm = np.linalg.norm(force_vec)
        if total_norm > max_force:
            force_vec = (force_vec / total_norm) * max_force

        return force_vec

    def plan(self, start_pos, goal_pos):
        """
        Executes a global path-planning query using the A* algorithm.

        Converts the 3D coordinates to grid indices, generates a binary occupancy 
        mask based on the UAV's current altitude, and performs the search. The 
        resulting path is smoothed before being returned in world coordinates.

        Args:
            start_pos (np.ndarray): Starting position $[x, y, z]$.
            goal_pos (np.ndarray): Target destination $[x, y, z]$.

        Returns:
            list[np.ndarray]|None: A list of 3D waypoints if a path is found, else None.
        """
        # 1. Conversion of positions to indices
        sx, sy = self._pos_to_idx(start_pos)
        gx, gy = self._pos_to_idx(goal_pos)
        
        # 2. Creation of the Binary Grid
        drone_z = start_pos[2]
        # True = Free (Walkable), False = Swollen wall
        walkable_grid = (self.h_map < drone_z).astype(int) 
        
        # --- CRASH FIX: Transposition (.T) ---
        # The pathfinding library reads [y][x], numpy is [x][y]
        grid = Grid(matrix=walkable_grid.T)
        
        # 3. Verification of limits
        if not grid.inside(sx, sy) or not grid.inside(gx, gy):
            print("[A*] Error: Departure or arrival outside the map.")
            return None

        #4. Management of Departure/Arrival Nodes
        node_start = grid.node(sx, sy)
        node_end = grid.node(gx, gy)
        
        # Forces a walkable take-off (to take off even if you are in the safety zone)
        node_start.walkable = True 
        
        if not node_end.walkable:
            print(f"[A*] Unattainable target ")
            return None

        # 5. Path calculation
        path, runs = self.finder.find_path(node_start, node_end, grid)
        
        if not path or len(path) < 2:
            return None
            
        # 6. Smoothing and Conversion
        smoothed_nodes = self._smooth_path(path, grid)
        
        # --- ‘SUBSCRIPTABLE’ CORRECTION: Use of p.x and p.y ---
        world_path = [self._idx_to_pos(p.x, p.y, goal_pos[2]) for p in smoothed_nodes]
        return world_path

    def _smooth_path(self, path_nodes, grid):
        """
        Simplifies the raw A* path using a 'String Pulling' heuristic.

        Iteratively checks for direct lines of sight between distant nodes to 
        remove redundant waypoints and produce a more natural trajectory.

        Args:
            path_nodes (list): Sequence of GridNode objects from the A* solver.
            grid (Grid): The pathfinding grid instance used for collision checks.

        Returns:
            list: A reduced sequence of optimized waypoints.
        """
        if len(path_nodes) < 3: return path_nodes
        smoothed = [path_nodes[0]]
        curr_idx = 0
        
        while curr_idx < len(path_nodes) - 1:
            best_next = curr_idx + 1
            for i in range(len(path_nodes) - 1, curr_idx, -1):
                if self._line_of_sight(grid, path_nodes[curr_idx], path_nodes[i]):
                    best_next = i
                    break
            curr_idx = best_next
            smoothed.append(path_nodes[curr_idx])
            
        return smoothed

    def _line_of_sight(self, grid, node_a, node_b):
        """Performs a 2D grid-based visibility check using Bresenham's algorithm.

        Args:
            grid (Grid): The navigation grid.
            node_a (Node): Starting node.
            node_b (Node): Target node.

        Returns:
            bool: True if the straight-line path between nodes is walkable.
        """
        x0, y0 = node_a.x, node_a.y
        x1, y1 = node_b.x, node_b.y
        dx = abs(x1 - x0); dy = abs(y1 - y0)
        x = x0; y = y0
        n = 1 + dx + dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        dx *= 2; dy *= 2
        
        for _ in range(n):
            if not grid.node(x, y).walkable: return False
            if error > 0:
                x += x_inc; error -= dy
            else:
                y += y_inc; error += dx
        return True

    def _pos_to_idx(self, pos):
        """Transforms world coordinates (meters) into discrete grid indices.

        Args:
            pos (np.ndarray): World position $[x, y]$.

        Returns:
            tuple[int, int]: Corresponding grid indices $(idx\_x, idx\_y)$.
        """
        nx = int((pos[0] - self.min_x) / self.res)
        ny = int((pos[1] - self.min_y) / self.res)
        return max(0, min(nx, self.width-1)), max(0, min(ny, self.height-1))

    def _idx_to_pos(self, x, y, z):
        """Transforms discrete grid indices back into world coordinates (meters).

        Applies a half-cell offset to ensure the position is at the cell center.

        Args:
            x (int): Grid index along the X-axis.
            y (int): Grid index along the Y-axis.
            z (float): Altitude to be assigned to the world point.

        Returns:
            np.ndarray: 3D world position vector.
        """
        # Offset to center of the cell
        return np.array([
            x * self.res + self.min_x + (self.res / 2.0),
            y * self.res + self.min_y + (self.res / 2.0),
            z])