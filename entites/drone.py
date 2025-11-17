import pybullet as p
from entites.agent import Agent 

class Drone(Agent):

    def __init__(self, identifier, coord, angle, speed, weight, drag):

        super().__init__(identifier)

        self.type = "drone"
        self.coord=coord
        self.angle = angle
        self.speed = speed
        self.weight = weight
        self.drag = drag
        self.physics_client_id = physics_client_id
        self.body_id = None         # PyBullet ID une fois chargé

        self._init_physics()


        