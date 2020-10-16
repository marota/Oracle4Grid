class Action:

    def __init__(self, atomic_actions):
        self.atomic_actions = {}
        self.sub = None
        self.line = None
        #init rest

    def get_sub(self):
        return self.sub

    def get_line(self):
        return self.sub

    def get_atomic_actions(self):
        return self.atomic_actions

    def get_depth(self):
        return len(self.atomic_actions)

    def print(self):
        #return string representing action

    def to_grid_to_op(self):
        #return to grid2op format