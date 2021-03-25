from grid2op.Agent import BaseAgent
import numpy as np

class OracleAgent(BaseAgent):
    """
    OracleAgent which plays the optimal simulated path in the action transition graph in a scenario
    """
    action_path = []

    def __init__(self,
                 action_space,
                 action_path,
                 oracle_action_path=None,
                 **kwargs):
        BaseAgent.__init__(self, action_space)
        self.action_path = action_path.copy()
        #self.actions_left = self.action_path.copy()
        if(oracle_action_path):
            self.oracle_action_path=oracle_action_path.copy()
        else:
            self.oracle_action_path=None

    def act(self, observation, reward, done):
        action = self.action_path.pop(0)

        #check line reco in addition if some were deconnected not on purpose
        action_line_reco=self.check_reconnect_line(observation)
        if(action_line_reco!=self.action_space({})):
            action_combined=action_line_reco
            action_combined+=action
            if(self.action_space._is_legal(action_combined,observation._obs_env)):
                return action_combined

        return action

    def check_reconnect_line(self,observation):

        res = {}  # add the do nothing
        line_stat_s = observation.line_status
        cooldown = observation.time_before_cooldown_line
        can_be_reco = ~line_stat_s & (cooldown == 0)

        # don't reconnect lines on which we played on purpose
        if(self.oracle_action_path is not None):
            oracle_action=self.oracle_action_path.pop(0)
            lines_oracle_action=oracle_action.lines
            if(len(lines_oracle_action)>=1):
                for l in lines_oracle_action:
                    can_be_reco[l]=False

        #reconnect useful lines that were in maintenance or attacked
        if np.any(can_be_reco):
            res = {"set_line_status": [(id_, +1) for id_ in np.where(can_be_reco)[0]]}
        return self.action_space(res)

    def reset(self, observation):
        #self.actions_left = self.action_path.copy()
        pass

    def load(self, path):
        # Nothing to load
        pass

    def save(self, path):
        # Nothing to save
        pass

    #@classmethod
    #def gen_next(cls, action_path):
    #    """
    #    This function allows to change the list of action dictionaries that the agent will perform.

    #    See the class level documentation for an example on how to use this.

    #    """
    #    cls.action_path = action_path.copy()
    #    return cls