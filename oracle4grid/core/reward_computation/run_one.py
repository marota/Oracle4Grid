# outputs a dataframe with rewards for each timestep for one run
import os

from grid2op.Agent import OneChangeThenNothing
from grid2op.Runner import Runner
from grid2op.Environment import Environment
from oracle4grid.core.reward_computation.Run import Run
from oracle4grid.core.utils.Action import OracleAction
from oracle4grid.core.agent.OneChangeThenOnlyReconnect import OneChangeThenOnlyReconnect
from oracle4grid.core.utils.constants import ENV_SEEDS, AGENT_SEEDS


def run_one(action: OracleAction, params_for_runner, max_iter: int,agent_seed,env_seed,path_logs=None):
    agent_class = OneChangeThenOnlyReconnect.gen_next(action.grid2op_action)
    runner = Runner(**params_for_runner, agentClass=agent_class)
    res = runner.run(nb_episode=1,
                              nb_process=1,
                              max_iter=max_iter, add_detailed_output=True,
                              env_seeds=env_seed,#ENV_SEEDS,
                              agent_seeds=agent_seed)#AGENT_SEEDS,
    return Run(action, res)
