# outputs a dataframe with rewards for each timestep for one run
import os

from grid2op.Agent import OneChangeThenNothing
from grid2op.Runner import Runner


def run_one(action, env, pbar, max_iter, nb_process):
    agent_class = OneChangeThenNothing.gen_next(action)
    runner = Runner(**env.get_params_for_runner(), agentClass=agent_class)
    res = runner.run(nb_episode=1,
                     nb_process=nb_process,
                     max_iter=max_iter,
                     )
    pbar.update(1)
    return res
