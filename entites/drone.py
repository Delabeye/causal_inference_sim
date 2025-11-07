from entites.agent import Agent 

class Drone(Agent):

    def __init__(self, identifier, coord, angle, speed, weight, drag):

        Agent.__init__(self, identifier)

        self.type = "drone"
        self.coord=coord
        self.angle = angle
        self.speed = speed
        self.weight = weight
        self.drag = drag

    
        