class Action:

    def __init__(self, atomic_actions):
        self.atomic_actions = atomic_actions
        self.subs = None
        self.lines = None
        #init rest

    def get_subs(self):
        return self.sub

    def get_lines(self):
        return self.sub

    def get_atomic_actions(self):
        return self.atomic_actions

    def get_depth(self):
        return len(self.atomic_actions)

    def print(self):
        # return string representing action

    def to_grid_to_op(self):
        # return to grid2op format