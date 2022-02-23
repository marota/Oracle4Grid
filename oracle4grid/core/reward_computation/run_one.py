# outputs a dataframe with rewards for each timestep for one run
import os

from grid2op.Agent import OneChangeThenNothing
from grid2op.Runner import Runner
from grid2op.Environment import Environment
from oracle4grid.core.reward_computation.Run import Run
from oracle4grid.core.utils.Action import OracleAction
from oracle4grid.core.agent.OneChangeThenOnlyReconnect import OneChangeThenOnlyReconnect
from oracle4grid.core.utils.prepare_environment import create_env_late_start_multivers
import warnings
from oracle4grid.core.utils.constants import ENV_SEEDS, AGENT_SEEDS


def run_one(action, params_for_runner, max_iter: int,agent_seed,env_seed,path_logs=None):
    if(type(action)==OracleAction):#we want to translate it to a grid2op action then
        action_grid2op=action.grid2op_action
    else:
        action_grid2op=action
    agent_class = OneChangeThenOnlyReconnect.gen_next(action_grid2op)
    runner = Runner(**params_for_runner, agentClass=agent_class)
    res = runner.run(nb_episode=1,
                              nb_process=1,
                              max_iter=max_iter, add_detailed_output=True,
                              env_seeds=env_seed,#ENV_SEEDS,
                              agent_seeds=agent_seed)#AGENT_SEEDS,
    return Run(action, res)

def run_one_multiverse(env_ref,action_univers, attack, begin,end,agent_seed,env_seed,path_logs=None):

    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        # TODO what do i do if agent cannot do opponent action ?
        # Retrieve line that is attacked
        line_id = attack.as_dict()['set_line_status']["disconnected_id"][0]
        action = action_univers.grid2op_action
        action.line_or_set_bus = [(line_id, 0)]
        action.line_ex_set_bus = [(line_id, 0)]
        if action.line_or_change_bus[line_id]:
            action.line_or_change_bus = [line_id]
        if action.line_ex_change_bus[line_id]:
            action.line_ex_change_bus = [line_id]
        combinated_action = action + attack

    env_universe = create_env_late_start_multivers(env_ref, begin + 1, end + 1)

    run=run_one(combinated_action, env_universe.get_params_for_runner(), end - begin, agent_seed, env_seed)
    #run_one(combinated_action, env.get_params_for_runner(), end - begin, [agent_seed], [env_seed])
    run.begin_ts=begin + 1
    run.action = action_univers
    run.attacks = [attack for i in range(begin,end+1)]
    run.max_ts = end + 1
    run.reset_attacks_id()


    return run


