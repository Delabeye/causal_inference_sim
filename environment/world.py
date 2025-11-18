# environment/world.py
import pybullet as p
import pybullet_data
import entities.obstacles as obs


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
