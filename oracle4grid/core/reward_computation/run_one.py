# outputs a dataframe with rewards for each timestep for one run
import os

from grid2op.Agent import OneChangeThenNothing
from grid2op.Runner import Runner
from grid2op.Environment import Environment
from oracle4grid.core.utils.Action import OracleAction


def run_one(action: OracleAction, env: Environment, max_iter: int):
    agent_class = OneChangeThenNothing.gen_next(action.grid2op_action_dict)
    runner = Runner(**env.get_params_for_runner(), agentClass=agent_class)
    res = runner.run_detailed(nb_episode=1,
                              nb_process=1,
                              max_iter=max_iter,
                              )
    return Run(action, res)


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
