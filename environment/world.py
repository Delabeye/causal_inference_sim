import pybullet as p
import pybullet_data
import entites.obstacles as obs # Nécessaire pour les types d'obstacles 

# Les imports pour 'drone' et 'sensor' sont supprimés car cette
# classe ne doit pas être responsable de leur création.

class World:
    def __init__(self, physics_client_id):
        self.p = p
        self.physics_client_id = physics_client_id
        # CORRECTION : Initialisé comme une liste vide
        self.obstacle_ids = []

    def load_basic_environment(self):
        """Charge le plan de base et définit la physique par défaut."""
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath(), 
                                         physicsClientId=self.physics_client_id)
        
        # Définir la gravité et le pas de temps
        self.p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client_id)
        self.p.setRealTimeSimulation(0, physicsClientId=self.physics_client_id) # Pas manuel
        
        # CORRECTION : Syntaxe corrigée (ajout de la position de base )
        self.p.loadURDF("plane.urdf",useFixedBase=1, 
                          physicsClientId=self.physics_client_id)
        
    # --- MÉTHODES AJOUTÉES POUR LES OBSTACLES PHYSIQUES ---

    def add_cube_obstacle(self, cube_obstacle: obs.CubeObstacle):
        """
        Crée un objet physique statique à partir d'une définition CubeObstacle.
        Utilise p.GEOM_BOX, qui nécessite 'halfExtents' (demi-étendues).[2, 6, 7]
        """
        half_extents = [cube_obstacle.length / 2, 
                        cube_obstacle.width / 2, 
                        cube_obstacle.height / 2] 
        
        # 1. Créer la forme de collision
        collision_shape_id = self.p.createCollisionShape(
            shapeType=self.p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.physics_client_id
        ) [2, 6, 5, 7, 8, 9]
        
        # 2. Créer la forme visuelle (peut être la même)
        visual_shape_id = self.p.createVisualShape(
            shapeType=self.p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.6, 0.6, 0.6, 1.0], # Couleur grise
            physicsClientId=self.physics_client_id
        )

        # 3. Créer le corps (MultiBody)
        # baseMass=0 rend l'objet statique 
        body_id = self.p.createMultiBody(
            baseMass=0, # STATIQUE
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=cube_obstacle.center,
            physicsClientId=self.physics_client_id
        ) [2, 3, 4, 5]
        self.obstacle_ids.append(body_id)
        return body_id

    def add_sphere_obstacle(self, sphere_obstacle: obs.SphericalObstacle):
        """
        Crée un objet physique statique à partir d'une définition SphericalObstacle.
        Utilise p.GEOM_SPHERE.[2, 6, 5, 7]
        """
        # 1. Créer la forme de collision
        collision_shape_id = self.p.createCollisionShape(
            shapeType=self.p.GEOM_SPHERE,
            radius=sphere_obstacle.radius,
            physicsClientId=self.physics_client_id
        ) [2, 6, 5, 7, 8, 9, 10, 11, 12]
        
        # 2. Créer la forme visuelle
        visual_shape_id = self.p.createVisualShape(
            shapeType=self.p.GEOM_SPHERE,
            radius=sphere_obstacle.radius,
            rgbaColor=[0.6, 0.6, 0.6, 1.0],
            physicsClientId=self.physics_client_id
        )

        # 3. Créer le corps (MultiBody)
        body_id = self.p.createMultiBody(
            baseMass=0, # STATIQUE 
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=sphere_obstacle.center,
            physicsClientId=self.physics_client_id
        ) [2, 3, 4, 5]
        self.obstacle_ids.append(body_id)
        return body_id

    def add_cylindrical_obstacle(self, cylindrical_obstacle: obs.CylindricalObstacle):
        """
        Crée un objet physique statique à partir d'une définition CylindricalObstacle.
        Utilise p.GEOM_CYLINDER. [2, 6, 7]
        """
        radius = cylindrical_obstacle.radius
        height = cylindrical_obstacle.height
        
        # 1. Créer la forme de collision
        collision_shape_id = self.p.createCollisionShape(
            shapeType=self.p.GEOM_CYLINDER,
            radius=radius,
            height=height, # PyBullet utilise 'height' ou 'length' [2, 6, 7]
            physicsClientId=self.physics_client_id
        )
        
        # 2. Créer la forme visuelle
        visual_shape_id = self.p.createVisualShape(
            shapeType=self.p.GEOM_CYLINDER,
            radius=radius,
            length=height, # La forme visuelle utilise 'length'
            rgbaColor=[0.6, 0.6, 0.6, 1.0],
            physicsClientId=self.physics_client_id
        )

        # 3. Créer le corps (MultiBody)
        body_id = self.p.createMultiBody(
            baseMass=0, # STATIQUE 
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=cylindrical_obstacle.center,
            physicsClientId=self.physics_client_id
        ) 
        self.obstacle_ids.append(body_id)
        return body_id

