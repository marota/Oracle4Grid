class Action:

    def __init__(self, atomic_actions):
        self.atomic_actions = atomic_actions
        self.subs = self.compute_subs()
        self.lines = self.compute_lines()
        #init rest

    def compute_subs(self):
        subs = {int(atomic_action[0].split('_')[1]) for atomic_action in self.atomic_actions if 'sub_' in atomic_action[0]}
        return subs

    def compute_lines(self):
        # TODO
        return

    def get_subs(self):
        return self.subs

    def get_lines(self):
        return self.subs

    def get_atomic_actions(self):
        return self.atomic_actions

    def get_depth(self):
        return len(self.atomic_actions)

    def print(self):
        # return string representing action
        return
    def to_grid_to_op(self):
        # return to grid2op format
        return