import numpy as np
import random
from scipy.interpolate import splprep, splev

class RRTNode:
    def __init__(self, pos, parent=None):
        self.pos = np.array(pos, dtype=float)
        self.parent = parent
        self.cost = 0.0

class RRTStarPlanner:
    def __init__(self, config: dict):
        self.bounds = config.get("world_bounds") or config.get("bounds")
        if not self.bounds:
            raise ValueError("Config must include 'world_bounds'")

        self.step_size = config.get("step_size", 1.0)
        self.max_iter = config.get("max_iter", 1000)
        self.safe_distance = config.get("safe_distance", 0.8)
        
        # On vise l'objectif directement 30% du temps pour aller plus vite
        self.goal_sample_rate = config.get("goal_sample_rate", 0.30) 
        self.search_radius = config.get("search_radius", 3.0)

    # -------------------------------------------------------------
    # Utilitaires Géométriques & Sampling
    # -------------------------------------------------------------
    def _distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def _random_point(self):
        """ Génère un point aléatoire uniforme. """
        # CORRECTION SYNTAXE ICI (plus d'étoiles *)
        return np.array([
            random.uniform(self.bounds["x"][0], self.bounds["x"][1]),
            random.uniform(self.bounds["y"][0], self.bounds["y"][1]),
            random.uniform(self.bounds["z"][0], self.bounds["z"][1]),
        ])

    def _random_point_informed(self, start, goal):
        """ 
        Stratégie "Informed RRT" : 
        Tire des points dans un cylindre bruité autour de l'axe Start->Goal.
        """
        # 30% du temps on explore partout pour éviter les culs-de-sac en U
        if random.random() < 0.3:
            return self._random_point()
        
        # Le reste du temps : Biais vers le but
        r = random.random()
        center = start + (goal - start) * r
        
        # Bruit gaussien (plus faible en Z pour rester à hauteur réaliste)
        noise = np.random.normal(0, 2.0, 3) 
        noise[2] *= 0.2 
        
        sample = center + noise
        
        # Clip pour rester dans les limites du monde
        sample[0] = np.clip(sample[0], self.bounds["x"][0], self.bounds["x"][1])
        sample[1] = np.clip(sample[1], self.bounds["y"][0], self.bounds["y"][1])
        sample[2] = np.clip(sample[2], self.bounds["z"][0], self.bounds["z"][1])
        
        return sample

    def _steer(self, from_node, to_point):
        from_pos = from_node.pos
        to_point = np.array(to_point, dtype=float)
        dist = self._distance(from_pos, to_point)
        if dist < self.step_size:
            return to_point
        direction = (to_point - from_pos) / (dist + 1e-9)
        return from_pos + direction * self.step_size

    # -------------------------------------------------------------
    # Vérifications de Collision (AVEC custom_margin)
    # -------------------------------------------------------------
    def _is_collision_free(self, pos, obstacles, custom_margin=None):
        """Vérifie la collision d'un point avec marge personnalisable."""
        margin = custom_margin if custom_margin is not None else self.safe_distance
        pos = np.array(pos)
        for ob in obstacles:
            if np.linalg.norm(pos - ob) < margin:
                return False
        return True

    def _is_path_collision_free(self, start, end, obstacles, exclude_start=False, custom_margin=None):
        """Vérifie la collision d'un segment avec marge personnalisable."""
        dist = self._distance(start, end)
        margin = custom_margin if custom_margin is not None else self.safe_distance
        
        if dist < 1e-6:
            return True if exclude_start else self._is_collision_free(start, obstacles, margin)

        # On échantillonne le segment (pas de 0.5 * margin)
        step_len = margin * 0.5
        steps = int(np.ceil(dist / step_len))
        start_idx = 1 if exclude_start else 0
        
        for i in range(start_idx, steps + 1):
            t = i / steps
            p = start + (end - start) * t
            if not self._is_collision_free(p, obstacles, margin):
                return False
        return True

    # -------------------------------------------------------------
    # RRT* Core
    # -------------------------------------------------------------
    def _find_near_nodes(self, nodes, new_node):
        return [n for n in nodes if self._distance(n.pos, new_node.pos) < self.search_radius]

    def _choose_parent(self, near_nodes, new_node, obstacles, start_in_collision=False):
        best_node = new_node.parent
        min_cost = new_node.cost
        for node in near_nodes:
            cost_via = node.cost + self._distance(node.pos, new_node.pos)
            if cost_via < min_cost:
                # Si le parent est la racine (départ) et qu'elle est en collision, on tolère le début du segment
                ignore_start = (node.parent is None and start_in_collision)
                if self._is_path_collision_free(node.pos, new_node.pos, obstacles, exclude_start=ignore_start):
                    min_cost = cost_via
                    best_node = node
        new_node.cost = min_cost
        new_node.parent = best_node
        return new_node

    def _rewire(self, nodes, near_nodes, new_node, obstacles):
        for node in near_nodes:
            if node.parent is None: continue
            new_cost = new_node.cost + self._distance(new_node.pos, node.pos)
            if new_cost < node.cost:
                if self._is_path_collision_free(new_node.pos, node.pos, obstacles):
                    node.parent = new_node
                    node.cost = new_cost

    def _generate_final_course(self, goal_node):
        path = []
        node = goal_node
        while node is not None:
            path.append(node.pos)
            node = node.parent
        return path[::-1]

    # -------------------------------------------------------------
    # Post-Processing
    # -------------------------------------------------------------
    def _resample_path(self, path, min_dist=0.5):
        """Simplifie le chemin pour avoir des points espacés régulièrement."""
        if len(path) < 2: return path
        new_path = [path[0]]
        last_added = path[0]
        
        for i in range(1, len(path) - 1):
            if np.linalg.norm(np.array(path[i]) - np.array(last_added)) >= min_dist:
                new_path.append(path[i])
                last_added = path[i]
                
        if np.linalg.norm(np.array(path[-1]) - np.array(last_added)) > 0.1:
            new_path.append(path[-1])
        else:
            new_path[-1] = path[-1] # Remplace le dernier si trop proche
        return new_path

    def smooth_path(self, path, smooth_factor=0.5):
        if len(path) < 3: return path
        try:
            path_np = np.array(path)
            unique = [path_np[0]]
            for i in range(1, len(path_np)):
                if np.linalg.norm(path_np[i] - path_np[i-1]) > 0.01:
                    unique.append(path_np[i])
            if len(unique) < 3: return path
            
            tck, u = splprep(np.array(unique).T, s=smooth_factor, k=min(3, len(unique)-1))
            u_new = np.linspace(0, 1, num=len(unique)*5)
            new_points = splev(u_new, tck)
            dense_path = np.array(new_points).T.tolist()
            
            return self._resample_path(dense_path, min_dist=0.5)
        except:
            return path

    # -------------------------------------------------------------
    # PLAN (Main)
    # -------------------------------------------------------------
    def plan(self, start, goal, obstacles, smooth=True):
        start = np.array(start)
        goal = np.array(goal)
        obs_np = [np.array(o) for o in (obstacles or [])]

        start_in_collision = not self._is_collision_free(start, obs_np)
        
        # Tolérance sur le but
        goal_margin = 0.5 
        
        start_node = RRTNode(start)
        node_list = [start_node]
        best_goal_node = None
        closest_node = start_node
        min_dist_goal = self._distance(start, goal)

        for _ in range(self.max_iter):
            # Sampling
            if random.random() < self.goal_sample_rate:
                rnd = goal
            else:
                rnd = self._random_point_informed(start, goal)

            # Nearest & Steer
            dists = [self._distance(n.pos, rnd) for n in node_list]
            nearest_node = node_list[np.argmin(dists)]
            new_pos = self._steer(nearest_node, rnd)

            # Checks
            if not self._is_collision_free(new_pos, obs_np):
                continue
            
            ignore_start = (nearest_node == start_node and start_in_collision)
            if not self._is_path_collision_free(nearest_node.pos, new_pos, obs_np, exclude_start=ignore_start):
                continue

            # Add Node
            new_node = RRTNode(new_pos, parent=nearest_node)
            new_node.cost = nearest_node.cost + self._distance(new_pos, nearest_node.pos)
            
            near_nodes = self._find_near_nodes(node_list, new_node)
            new_node = self._choose_parent(near_nodes, new_node, obs_np, start_in_collision)
            node_list.append(new_node)
            self._rewire(node_list, near_nodes, new_node, obs_np)

            # Goal Check
            dist_to_goal = self._distance(new_node.pos, goal)
            if dist_to_goal < min_dist_goal:
                min_dist_goal = dist_to_goal
                closest_node = new_node

            if dist_to_goal <= self.step_size:
                # --- CORRECTION ICI : Utilisation de custom_margin ---
                if self._is_path_collision_free(new_node.pos, goal, obs_np, custom_margin=goal_margin):
                    final = RRTNode(goal, parent=new_node)
                    final.cost = new_node.cost + dist_to_goal
                    if best_goal_node is None or final.cost < best_goal_node.cost:
                        best_goal_node = final

        final = best_goal_node if best_goal_node else closest_node
        if final == start_node: return None

        path = self._generate_final_course(final)
        
        # Nettoyage
        if len(path) > 1 and np.linalg.norm(np.array(path[0]) - start) < 0.1:
            path.pop(0)
            
        return self.smooth_path(path) if smooth else path