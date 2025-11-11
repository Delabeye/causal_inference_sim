class Entites:

    def __init__(self, identifier):
        self.identifier = identifier

    def __hash__(self):
        return hash(self.identifier)

