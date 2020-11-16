from oracle4grid.core.utils.Action import OracleAction


class Run:
    def __init__(self, action: OracleAction, id_chron, name_chron, cum_reward, nb_timestep, max_ts, rewards):
        """

        :type action: oracle4grid.core.utils.Action
        :type res: list
        """
        self.action = action
        # We should always only have one res (because we only use one chronic)
        self.id_chron = id_chron
        self.name_chron = name_chron
        self.cum_reward = cum_reward
        self.nb_timestep = nb_timestep
        self.max_ts = max_ts
        self.rewards = rewards
