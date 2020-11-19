import numpy
from grid2op.Agent.BaseAgent import BaseAgent


class OneChangeThenOnlyReconnect(BaseAgent):
    my_dict = {}

    def __init__(self, action_space):
        BaseAgent.__init__(self, action_space)
        self.has_changed = False

    def get_reconnect(self, observation):
        res = {}  # add the do nothing
        line_stat_s = observation.line_status
        cooldown = observation.time_before_cooldown_line
        can_be_reco = ~line_stat_s & (cooldown == 0)
        if numpy.any(can_be_reco):
            res = {"set_line_status": [(id_, +1) for id_ in numpy.where(can_be_reco)[0]]}
        return self.action_space(res)

    def act(self, observation, reward, done=False):
        if self.has_changed:
            res = self.get_reconnect(observation)
        else:
            res = self.action_space(self._get_dict_act())
            self.has_changed = True
        return res

    def reset(self, obs):
        self.has_changed = False

    def _get_dict_act(self):
        """
        Function that need to be overridden to indicate which action to perform.

        Returns
        -------
        res: ``dict``
            A dictionnary that can be converted into a valid :class:`grid2op.BaseAction.BaseAction`. See the help of
            :func:`grid2op.BaseAction.ActionSpace.__call__` for more information.
        """
        return self.my_dict

    @classmethod
    def gen_next(cls, dict_):
        """
        This function allows to change the dictionnary of the action that the agent will perform.

        See the class level documentation for an example on how to use this.

        Parameters
        ----------
        dict_: ``dict``
            A dictionnary representing an action. This dictionnary is assumed to be convertible into an action.
            No check is performed at this stage.


        """
        cls.my_dict = dict_
        return cls