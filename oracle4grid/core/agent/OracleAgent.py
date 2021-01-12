from grid2op.Agent import BaseAgent

class OracleAgent(BaseAgent):
    """
    OracleAgent which plays the optimal simulated path in the action transition graph in a scenario
    """
    def __init__(self,
                 action_path,
                 action_space,
                 observation_space,
                 name,
                 **kwargs):
        super().__init__(action_space)
        self.name = name
        self.action_path = action_path.copy()
        self.actions_left = action_path.copy()

    def act(self, observation, reward, done):
        action_dict = self.actions_left.pop(0)
        return self.action_space(action_dict)

    def reset(self, observation):
        self.actions_left = self.action_path.copy()
        pass

    def load(self, path):
        # Nothing to load
        pass

    def save(self, path):
        # Nothing to save
        pass

    @classmethod
    def gen_next(cls, action_path):
        """
        This function allows to change the list of action dictionaries that the agent will perform.

        See the class level documentation for an example on how to use this.

        """
        cls.action_path = action_path
        return cls