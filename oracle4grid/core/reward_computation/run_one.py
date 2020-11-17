# outputs a dataframe with rewards for each timestep for one run
import os

from grid2op.Agent import OneChangeThenNothing
from grid2op.Runner import Runner
from grid2op.Environment import Environment
from oracle4grid.core.reward_computation.Run import Run
from oracle4grid.core.utils.Action import OracleAction


def run_one(action: OracleAction, env: Environment, max_iter: int):
    agent_class = OneChangeThenNothing.gen_next(action.grid2op_action_dict)
    runner = Runner(**env.get_params_for_runner(), agentClass=agent_class)
    res = runner.run_detailed(nb_episode=1,
                              nb_process=1,
                              max_iter=max_iter, force_detail=True
                              )
    return Run(action, res)
