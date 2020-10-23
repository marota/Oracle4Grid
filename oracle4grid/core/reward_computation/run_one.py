# outputs a dataframe with rewards for each timestep for one run
import os

from grid2op.Agent import OneChangeThenNothing
from grid2op.Runner import Runner


def run_one(action, env, max_iter):
    agent_class = OneChangeThenNothing.gen_next(action)
    runner = Runner(**env.get_params_for_runner(), agentClass=agent_class)
    res = runner.run_detailed(nb_episode=1,
                              nb_process=1,
                              max_iter=max_iter,
                              )
    return Run(action, res)


class Run:
    def __init__(self, action, res):
        """

        :type action: oracle4grid.core.utils.Action
        :type res: list
        """
        self.action = action
        i, cum_reward, nb_timestep, episode_data = res.pop()
        self.id = i
        self.cum_reward = cum_reward
        self.nb_timestep = nb_timestep
        # TODO : Change that to df with timestep and reward
        self.episode_data = episode_data