import warnings
from math import isclose
import os

from grid2op.Parameters import Parameters
from grid2op.Runner import Runner


from oracle4grid.core.agent.OracleAgent import OracleAgent
from oracle4grid.core.utils.constants import EnvConstants
from oracle4grid.core.utils.prepare_environment import prepare_env, get_initial_configuration

def replay(action_path: list, max_iter: int,
           kpis, grid_path, chronic_scenario, debug = False, constants=EnvConstants(), env_seed = None, agent_seed = None,
           rel_tol = 1e7, path_logs = None, logs_file_name_extension=None,oracle_action_path=None):
    if debug:
        print('\n')
        print("============== 6 - Replay OracleAgent on best path with real condition game rules ==============")
    # Environment settings for replay
    param = Parameters()
    param.init_from_dict(constants.DICT_GAME_PARAMETERS_REPLAY)
    env, chronic_id = prepare_env(grid_path, chronic_scenario, param, constants=constants)
    init_topo_vect, init_line_status = get_initial_configuration(env)
    env.set_id(chronic_id)


    # Run replay
    ##############
    # With runner - OK but some differences in decimals
    if path_logs is not None:
        if logs_file_name_extension is not None:
            path_logs=os.path.join(path_logs, "replay_logs_"+ logs_file_name_extension)
        else:
            path_logs = os.path.join(path_logs, "replay_logs")

    #agent_class = OracleAgent(env.action_space, oracle_action_path).gen_next(action_path)
    agent = OracleAgent(env.action_space,action_path, oracle_action_path,
                        init_topo_vect=init_topo_vect,init_line_status=init_line_status)#.gen_next(action_path)
    if agent_seed is not None:  # Grid2op runner expect a list of seeds
        agent_seed=[agent_seed]
    if env_seed is not None:
        env_seed=[env_seed]
    runner = Runner(**env.get_params_for_runner(), agentClass=None,agentInstance=agent)

    name_chron, agent_reward, nb_time_step, episode_data =runner.run_one_episode(indx=chronic_id,
                             path_save=path_logs,
                             pbar=True,
                             env_seed=env_seed,  # ENV_SEEDS,
                             max_iter=max_iter,
                             agent_seed=agent_seed,  # AGENT_SEEDS,
                             detailed_output=True)
    #res = runner.run(nb_episode=1,
    #                 nb_process=1,
    #                 max_iter=max_iter, add_detailed_output=True,
    #                 env_seeds=env_seed,  # ENV_SEEDS,
    #                 agent_seeds=agent_seed,  # AGENT_SEEDS,
    #                 path_save=path_logs
    #                 )
    #id_chron, name_chron, agent_reward, t, max_ts, episode_data = res.pop()


    # Check reward as expected
    expected_reward = extract_expected_reward(kpis)
    if (not isclose(expected_reward, agent_reward, rel_tol=rel_tol)) and nb_time_step==max_iter:
        warnings.warn("During replay - oracle agent does not retrieve the expected reward. Some timestep may have break some game rules in real condition. Expected reward: "+str(expected_reward)+" Reward obtained: "+str(agent_reward))
    elif nb_time_step==max_iter: # if pas de game over
        print("Expected reward of "+str(expected_reward)+" has been correctly obtained in replay conditions")
    else:
        warnings.warn("During replay - oracle agent has game over before max iter (timestep " + str(nb_time_step) + ")")
    return nb_time_step

def extract_expected_reward(kpis):
    return kpis.loc[kpis['Indicator name']=="Best possible path with game rules", "Reward value"].values[0]