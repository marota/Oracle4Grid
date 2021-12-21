import numpy
from grid2op.Agent import BaseAgent


class OneChangeThenOnlyReconnect(BaseAgent):
    my_action = None

    def __init__(self, action_space):
        BaseAgent.__init__(self, action_space)
        self.has_changed = False

         # Check if action concerns a line
        impact = self._get_act().impact_on_objects()

        # Find ids of untouchable lines because they are disconnected on purpose and sould not be disconnected
        self.untouchable_line_ids = impact['force_line']['disconnections']['powerlines']
        if len(self.untouchable_line_ids)>0:
            self.untouchable_line_ids = self.untouchable_line_ids.tolist()
        self.touchable_line_id_vec = numpy.array([i not in self.untouchable_line_ids for i in range(self.action_space.n_line)])

    def get_reconnect(self, observation):
        res = {}  # add the do nothing
        line_stat_s = observation.line_status
        cooldown = observation.time_before_cooldown_line
        can_be_reco = ~line_stat_s & (cooldown == 0) & self.touchable_line_id_vec
        if numpy.any(can_be_reco):
            res = {"set_line_status": [(id_, +1) for id_ in numpy.where(can_be_reco)[0]]}
        return self.action_space(res)

    def act(self, observation, reward, done=False):
        if self.has_changed:
            res = self.get_reconnect(observation)
        else:
            res = self._get_act()
            self.has_changed = True
        return res

    def reset(self, obs):
        self.has_changed = False

    def _get_act(self):
        """
        Function that need to be overridden to indicate which action to perform.

        Returns
        -------
        res: ``BaseAction``
            An action of :class:`grid2op.BaseAction.BaseAction`. See the help of
            :func:`grid2op.BaseAction.ActionSpace.__call__` for more information.
        """
        return self.my_action

    @classmethod
    def gen_next(cls, action_):
        """
        This function allows to change the dictionnary of the action that the agent will perform.

        See the class level documentation for an example on how to use this.

        Parameters
        ----------
        dict_: ``dict``
            A dictionnary representing an action. This dictionnary is assumed to be convertible into an action.
            No check is performed at this stage.


        """
        cls.my_action = action_
        return cls