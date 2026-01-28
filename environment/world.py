# environment/world.py
import pybullet as p
import pybullet_data
import entities.obstacles as obs
import random
import os

class World:
    """
    World class for managing physics simulation environment and city generation.
    This class initializes a PyBullet physics client and provides utilities for
    generating procedural city environments with buildings and obstacles.
    Attributes:
        p: PyBullet module reference for physics operations.
        physics_client_id (int): Unique identifier for the PyBullet physics client.
        obstacle_ids (list[int]): List of PyBullet body IDs for obstacles in the world.
    Methods:
        __init__(physics_client_id: int) -> None:
            Initializes the World with a physics client, sets gravity, and configures
            the simulation environment.
        generate_city_urdf(config: dict) -> list[dict]:
            Generates a procedural city layout and creates a URDF file with buildings.
            Args:
                config (dict): Configuration dictionary with optional keys:
                    - filename (str): Name of the URDF file to generate. Default: "city"
                    - n_blocks_x (int): Number of city blocks along X-axis. Default: 4
                    - n_blocks_y (int): Number of city blocks along Y-axis. Default: 4
                    - block_size (float): Size of each block in meters. Default: 20.0
                    - road_width (float): Width of roads between blocks in meters. Default: 4.0
                    - buildings_per_side (int): Number of buildings per side within a block. Default: 3
            Returns:
                list[dict]: List of building dictionaries, each containing:
                    - id (int): Unique building identifier
                    - center (list[float]): [x, y, height] coordinates of building center
                    - height (float): Height of the building in meters
                    - width (float): Width of the building in meters
                    - length (float): Depth/length of the building in meters
            The generated URDF file includes:
            - Ground plane (asphalt) as the base world link
            - Procedurally generated buildings with random heights (30-50m) and colors
            - Fixed joints attaching all buildings to the ground plane
            - Inertial, visual, and collision properties for each building
    """
    def __init__(self, physics_client_id: int):
        """
        Initializes the simulation world and configures global physics parameters.

        Sets the standard gravity vector, disables real-time simulation to ensure 
        deterministic behavior, and configures internal PyBullet search paths for 
        standard assets.

        Args:
            physics_client_id (int): The handle returned by `p.connect()`.
        """
        self.p = p
        self.physics_client_id = physics_client_id
        self.obstacle_ids: list[int] = []
        self.p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client_id)
        self.p.setRealTimeSimulation(0, physicsClientId=self.physics_client_id)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
    def generate_city_urdf(self,config):
        """
        Generates a procedural city environment as a single multi-link URDF file.

        This method calculates a grid-based city layout consisting of blocks, roads, 
        and buildings. It writes a XML-based URDF file where each building is 
        modeled as a rigid link attached to a central ground plane. The buildings 
        feature randomized heights and visual properties (colors) while maintaining 
        precise collision geometries and inertial tensors for physics stability.

        Args:
            config (dict): Configuration parameters for the generator.
            - filename (str): Target URDF filename (saved in 'assets/').
            - n_blocks_x (int): Number of grid divisions along the X-axis.
            - n_blocks_y (int): Number of grid divisions along the Y-axis.
            - block_size (float): Dimension of a single square city block (m).
            - road_width (float): Clearance between city blocks (m).
            - buildings_per_side (int): Building density within a single block.

        Returns:
            list[dict]: A list of metadata for every spawned building, including 
                geometric dimensions and absolute world coordinates for 
                use by path-planners or sensors.
                
        Note:
            Building heights are randomized to simulate an urban 'canyon' environment.
        """
    
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

        # --- BASE LINK (Ground) ---
        # --- GROUND (ASPHALT) ---
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
            # Spacing between block centers
            stride = block_size + road_width
            # Spacing between buildings inside a single block
            inner_spacing = block_size / buildings_per_side

            # --- BLOCK LOOP ---
            for bx in range(n_blocks_x):
                for by in range(n_blocks_y):
                    # Calculate the center of the block
                    block_center_x = (bx - n_blocks_x / 2.0) * stride + stride / 2.0
                    block_center_y = (by - n_blocks_y / 2.0) * stride + stride / 2.0

                    # --- LOOP THROUGH BUILDINGS WITHIN THE BLOCK ---
                    for ix in range(buildings_per_side):
                        for iy in range(buildings_per_side):
                        
                            # Random height (Towers)
                            h = round(random.uniform(30.0, 50.0), 1)
                            # Width adjusted to leave a small gap between buildings
                            w = inner_spacing * 0.9
                            d = inner_spacing * 0.9

                            # Position relative to the block center
                            rel_x = (ix - buildings_per_side / 2.0) * inner_spacing + inner_spacing / 2.0
                            rel_y = (iy - buildings_per_side / 2.0) * inner_spacing + inner_spacing / 2.0
                        
                            abs_x = block_center_x + rel_x
                            abs_y = block_center_y + rel_y
                            abs_z = h / 2.0
                        
                            name = f"bld_{building_id}"
                            color = f"{round(random.uniform(0.3, 0.6), 2)} {round(random.uniform(0.3, 0.6), 2)} {round(random.uniform(0.3, 0.6), 2)} 1"

                            buildings.append({
                            "id": building_id,
                            "center": [abs_x, abs_y, h],
                            "height": h,
                            "width": w,
                            "length": d, # Corrected spelling
                            })

                            mass = 100000.0  # Fixed or calculated mass (e.g., w * d * h * density)
                            ixx = (1/12.0) * mass * (d**2 + h**2)
                            iyy = (1/12.0) * mass * (w**2 + h**2)
                            izz = (1/12.0) * mass * (w**2 + d**2)

                            f.write(f'  <link name="{name}">\n')
                        
                            # --- INERTIA BLOCK ---
                            f.write('    <inertial>\n')
                            # The inertia origin is placed at the center of the link (0 0 0)
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

                            # The joint remains "fixed" so the building stays static
                            f.write(f'  <joint name="j_{name}" type="fixed">\n')
                            f.write('    <parent link="ground_plane"/>\n')
                            f.write(f'    <child link="{name}"/>\n')
                            f.write(f'    <origin xyz="{abs_x} {abs_y} {abs_z}" rpy="0 0 0"/>\n')
                            f.write('  </joint>\n\n')

                            building_id += 1

            f.write('</robot>\n')
        return buildings
