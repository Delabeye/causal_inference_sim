import numpy as np
from scipy.spatial import KDTree
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

class HeightmapAStar:
    def __init__(self, config, resolution=0.25):
        self.res = resolution
        self.bounds = config.get("world_bounds", {'x': [-500, 500], 'y': [-500, 500], 'z': [0, 100]})
        self.min_x, self.min_y = self.bounds['x'][0], self.bounds['y'][0]
        
        # Dimensions de la grille
        self.width = int((self.bounds['x'][1] - self.min_x) / self.res)
        self.height = int((self.bounds['y'][1] - self.min_y) / self.res)
        
        # Heightmap en float32
        self.h_map = np.zeros((self.width, self.height), dtype=np.float32)
        
        # Mouvement : Interdit de frôler les coins (Only when no obstacle)
        self.finder = AStarFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle)
        
        # Récupération de la marge en MÈTRES (ex: 1.0m)
        self.safety_margin = config.get("safety_margin", 1.0)

    def build_from_buildings(self, buildings):
        """ 
        Construit la carte en appliquant l'inflation GÉOMÉTRIQUE.
        On augmente la taille physique des bâtiments avant de les mettre sur la grille.
        """
        source_data = buildings.values() if isinstance(buildings, dict) else buildings
        centers = []
        
        for data in source_data:
            center = data["center"]
            centers.append(center[:2])
            
            # Dimensions réelles
            h = data["height"]
            real_w = data["width"]
            real_l = data["length"]
            
            # --- INFLATION GÉOMÉTRIQUE (La Solution) ---
            # On ajoute la marge de sécurité des deux côtés.
            # Si marge = 1m, l'immeuble devient 2m plus large et plus long.
            effective_w = real_w + (2.0 * self.safety_margin)
            effective_l = real_l + (2.0 * self.safety_margin)
            
            # Calcul des indices avec les dimensions gonflées
            # Axe X (Width)
            x_min = int((center[0] - effective_w/2 - self.min_x) / self.res)
            x_max = int((center[0] + effective_w/2 - self.min_x) / self.res)
            
            # Axe Y (Length)
            y_min = int((center[1] - effective_l/2 - self.min_y) / self.res)
            y_max = int((center[1] + effective_l/2 - self.min_y) / self.res)
            
            # Clamping pour ne pas sortir de la carte
            x0 = max(0, x_min); x1 = min(self.width, x_max)
            y0 = max(0, y_min); y1 = min(self.height, y_max)
            
            # Remplissage
            if x0 < x1 and y0 < y1:
                self.h_map[x0:x1, y0:y1] = np.maximum(self.h_map[x0:x1, y0:y1], h)
                
        if centers:
            self.building_tree = KDTree(np.array(centers))

    def plan(self, start_pos, goal_pos):
        # 1. Conversion positions -> indices
        sx, sy = self._pos_to_idx(start_pos)
        gx, gy = self._pos_to_idx(goal_pos)
        
        # 2. Création de la Grille Binaire
        drone_z = start_pos[2]
        # True = Libre (Walkable), False = Mur gonflé
        walkable_grid = (self.h_map < drone_z).astype(int) 
        
        # --- CORRECTION CRASH : Transposition (.T) ---
        # La librairie pathfinding lit [y][x], numpy est [x][y]
        grid = Grid(matrix=walkable_grid.T)
        
        # 3. Vérification des limites
        if not grid.inside(sx, sy) or not grid.inside(gx, gy):
            print("[A*] Erreur : Départ ou Arrivée hors de la carte.")
            return None

        # 4. Gestion des Nœuds Départ/Arrivée
        node_start = grid.node(sx, sy)
        node_end = grid.node(gx, gy)
        
        # Force le départ walkable (pour décoller même si on est dans la zone de sécurité)
        node_start.walkable = True 
        
        if not node_end.walkable:
            print(f"[A*] Cible inaccessible (dans un mur gonflé).")
            return None

        # 5. Calcul du chemin
        path, runs = self.finder.find_path(node_start, node_end, grid)
        
        if not path or len(path) < 2:
            return None
            
        # 6. Lissage et Conversion
        smoothed_nodes = self._smooth_path(path, grid)
        
        # --- CORRECTION "SUBSCRIPTABLE" : Utilisation de p.x et p.y ---
        world_path = [self._idx_to_pos(p.x, p.y, goal_pos[2]) for p in smoothed_nodes]
        return world_path

    def _smooth_path(self, path_nodes, grid):
        """ Simplification du chemin (String Pulling) """
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
        x0, y0 = node_a.x, node_a.y
        x1, y1 = node_b.x, node_b.y
        dx = abs(x1 - x0); dy = abs(y1 - y0)
        x = x0; y = y0
        n = 1 + dx + dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        dx *= 2; dy *= 2
        
        for i in range(n):
            if not grid.node(x, y).walkable: return False
            if error > 0:
                x += x_inc; error -= dy
            else:
                y += y_inc; error += dx
        return True

    def _pos_to_idx(self, pos):
        nx = int((pos[0] - self.min_x) / self.res)
        ny = int((pos[1] - self.min_y) / self.res)
        return max(0, min(nx, self.width-1)), max(0, min(ny, self.height-1))

    def _idx_to_pos(self, x, y, z):
        # Retourne le CENTRE de la case pour éloigner des murs
        return np.array([
            x * self.res + self.min_x + (self.res / 2.0),
            y * self.res + self.min_y + (self.res / 2.0),
            z])