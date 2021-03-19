from oracle4grid.core.utils.Action import OracleAction


class Run:
    def __init__(self, action: OracleAction, res: list):
        """

        :type action: oracle4grid.core.utils.Action
        :type res: list
        """
        self.action = action
        # We should always only have one res (because we only use one chronic)
        id_chron, name_chron, cum_reward, nb_timestep, max_ts, episode_data = res.pop()
        self.id_chron = id_chron
        self.name_chron = name_chron
        self.cum_reward = cum_reward
        self.nb_timestep = nb_timestep
        self.max_ts = max_ts
        self.rewards = episode_data.rewards
        self.other_rewards = episode_data.other_rewards
        self.attacks = episode_data.attacks
        self.attacks_id=[None if len(attack.impact_on_objects()['force_line']['disconnections']['powerlines']) == 0 else
         attack.impact_on_objects()['force_line']['disconnections']['powerlines'][0] for attack in self.attacks]
        #self.attacks_id=[run.attacks[t].impact_on_objects()['force_line']['disconnections']['powerlines'] for t, attack in self.attack]

