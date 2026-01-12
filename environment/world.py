# environment/world.py
import pybullet as p
import pybullet_data
import entities.obstacles as obs
from utilities.utilities import discretize_building

class World:
    def __init__(self, physics_client_id: int):
        self.p = p
        self.physics_client_id = physics_client_id
        self.obstacle_ids: list[int] = []

    def load_basic_environment(self) -> None:
        """Charge le sol + règle la physique."""
        # Chemin vers les assets PyBullet
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Physique de base
        self.p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client_id)
        self.p.setRealTimeSimulation(0, physicsClientId=self.physics_client_id)

        # Sol
        self.p.loadURDF(
            "plane.urdf",
            useFixedBase=1,
            physicsClientId=self.physics_client_id,
        )

    # ---------- OBSTACLES ----------

    def add_cube_obstacle(self, cube_obstacle: obs.CubeObstacle) -> int:
        half_extents = [
            cube_obstacle.length / 2.0,
            cube_obstacle.width / 2.0,
            cube_obstacle.height / 2.0,
        ]

        col_id = self.p.createCollisionShape(
            shapeType=self.p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.physics_client_id,
        )

        vis_id = self.p.createVisualShape(
            shapeType=self.p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[1.0, 0.0, 0.0, 1.0],  # ROUGE bien visible
            physicsClientId=self.physics_client_id,
        )

        body_id = self.p.createMultiBody(
            baseMass=0.0,  # statique
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=cube_obstacle.center,
            physicsClientId=self.physics_client_id,
        )
        self.obstacle_ids.append(body_id)
        return body_id

    def add_sphere_obstacle(self, sphere_obstacle: obs.SphericalObstacle) -> int:
        col_id = self.p.createCollisionShape(
            shapeType=self.p.GEOM_SPHERE,
            radius=sphere_obstacle.radius,
            physicsClientId=self.physics_client_id,
        )

        vis_id = self.p.createVisualShape(
            shapeType=self.p.GEOM_SPHERE,
            radius=sphere_obstacle.radius,
            rgbaColor=[0.0, 1.0, 0.0, 1.0],  # VERT bien visible
            physicsClientId=self.physics_client_id,
        )

        body_id = self.p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=sphere_obstacle.center,
            physicsClientId=self.physics_client_id,
        )
        self.obstacle_ids.append(body_id)
        return body_id

    def add_cylindrical_obstacle(self, cylindrical_obstacle: obs.CylindricalObstacle) -> int:
        radius = cylindrical_obstacle.radius
        height = cylindrical_obstacle.height

        # Collision
        col_id = self.p.createCollisionShape(
            shapeType=self.p.GEOM_CYLINDER,
            radius=radius,
            height=height,
            physicsClientId=self.physics_client_id,
        )

        # Visuel : pour le cylindre, le mot-clé est `length`
        vis_id = self.p.createVisualShape(
            shapeType=self.p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=[0.0, 0.0, 1.0, 1.0],  # BLEU bien visible
            physicsClientId=self.physics_client_id,
        )

        body_id = self.p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=cylindrical_obstacle.center,
            physicsClientId=self.physics_client_id,
        )
        self.obstacle_ids.append(body_id)
        return body_id


import random
import os
def generate_city_urdf(config,
                        res=0.5):
    
    filename=config.get("filename","city")
    n_blocks_x = config.get("n_blocks_x", 4)
    n_blocks_y = config.get("n_blocks_y", 4)
    block_size = config.get("block_size", 20.0)
    road_width = config.get("road_width", 4.0)

    buildings_per_side = config.get("buildings_per_side", 3)
    city_file = os.path.join("assets", f"{filename}.urdf")

    buildings = []

    with open(city_file, "w") as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write('<robot name="city">\n\n')

        # --- LIEN DE BASE (Le sol) ---
        # --- SOL (ASPHALTE) ---
        total_size_x = n_blocks_x * (block_size + road_width)
        total_size_y = n_blocks_y * (block_size + road_width)
        
        f.write('  <link name="world_link"/>\n')
        f.write('  <link name="ground_plane">\n')
        f.write('    <visual>\n')
        f.write(f'      <geometry><box size="{total_size_x} {total_size_y} 0.1"/></geometry>\n')
        f.write('      <material name="asphalt"><color rgba="0.1 0.1 0.1 1"/></material>\n')
        f.write('    </visual>\n')
        f.write('    <collision>\n')
        f.write(f'      <geometry><box size="{total_size_x} {total_size_y} 0.1"/></geometry>\n')
        f.write('    </collision>\n')
        f.write('  </link>\n\n')
        
        f.write('  <joint name="ground_joint" type="fixed">\n')
        f.write('    <parent link="world_link"/><child link="ground_plane"/>\n')
        f.write('  </joint>\n\n')

        building_id = 0
        # Espacement entre les centres des blocs
        stride = block_size + road_width
        # Espacement entre les immeubles à l'intérieur d'un bloc
        inner_spacing = block_size / buildings_per_side

        # --- BOUCLE DES BLOCS ---
        for bx in range(n_blocks_x):
            for by in range(n_blocks_y):
                # Calcul du centre du bloc
                block_center_x = (bx - n_blocks_x / 2.0) * stride + stride / 2.0
                block_center_y = (by - n_blocks_y / 2.0) * stride + stride / 2.0

                # --- BOUCLE DES IMMEUBLES DANS LE BLOC ---
                for ix in range(buildings_per_side):
                    for iy in range(buildings_per_side):
                        
                        # Hauteur aléatoire (Tours)
                        h = round(random.uniform(30.0, 50.0),1)
                        # Largeur ajustée pour laisser un petit espace entre immeubles
                        w = inner_spacing * 0.9
                        d = inner_spacing * 0.9
                        
                        # Position relative au centre du bloc
                        rel_x = (ix - buildings_per_side / 2.0) * inner_spacing + inner_spacing / 2.0
                        rel_y = (iy - buildings_per_side / 2.0) * inner_spacing + inner_spacing / 2.0
                        
                        abs_x = block_center_x + rel_x
                        abs_y = block_center_y + rel_y
                        abs_z = h / 2.0
                        
                        name = f"bld_{building_id}"
                        color = f"{round(random.uniform(0.3, 0.6),2)} {round(random.uniform(0.3, 0.6),2)} {round(random.uniform(0.3, 0.6),2)} 1"

                        buildings.append({
                            "id": building_id,
                            "center": [abs_x, abs_y, h],
                            "height": h,
                            "width": w,
                            "length": d, # Corrected spelling
                            })

                        mass = 100000.0  # Masse fixe ou calculée (ex: w * d * h * densité)
                        ixx = (1/12.0) * mass * (d**2 + h**2)
                        iyy = (1/12.0) * mass * (w**2 + h**2)
                        izz = (1/12.0) * mass * (w**2 + d**2)

                        f.write(f'  <link name="{name}">\n')
                        
                        # --- BLOC INERTIE ---
                        f.write('    <inertial>\n')
                        # On place l'origine de l'inertie au centre du lien (0 0 0)
                        f.write('      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
                        f.write(f'      <mass value="{mass}"/>\n')
                        f.write(f'      <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>\n')
                        f.write('    </inertial>\n')
                        
                        # --- VISUAL ---
                        f.write('    <visual>\n')
                        f.write(f'      <geometry><box size="{w} {d} {h}"/></geometry>\n')
                        f.write(f'      <material name="mat_{building_id}"><color rgba="{color}"/></material>\n')
                        f.write('    </visual>\n')
                        
                        # --- COLLISION ---
                        f.write('    <collision>\n')
                        f.write(f'      <geometry><box size="{w} {d} {h}"/></geometry>\n')
                        f.write('    </collision>\n')
                        f.write('  </link>\n')

                        # Le joint reste "fixed" pour que le bâtiment ne bouge pas
                        f.write(f'  <joint name="j_{name}" type="fixed">\n')
                        f.write('    <parent link="ground_plane"/>\n')
                        f.write(f'    <child link="{name}"/>\n')
                        f.write(f'    <origin xyz="{abs_x} {abs_y} {abs_z}" rpy="0 0 0"/>\n')
                        f.write('  </joint>\n\n')
                        
                        building_id += 1

        f.write('</robot>\n')
        return buildings
