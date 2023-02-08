import os
import time
import pandas as pd
import numpy as np

from grid2op.Episode import EpisodeData
import json
from grid2op.dtypes import dt_float, dt_bool
from oracle4grid.core.actions_utils import combinator
from oracle4grid.core.graph import graph_generator, compute_trajectory, indicators
from oracle4grid.core.replay import agent_replay
from oracle4grid.core.reward_computation import run_many
from oracle4grid.core.utils.launch_utils import OracleParser, load
from oracle4grid.core.utils.prepare_environment import get_initial_configuration
from oracle4grid.core.reward_computation.attacks_multiverse import multiverse_simulation
from oracle4grid.core.utils.config_ini_utils import MAX_ITER, MAX_DEPTH, NB_PROCESS, N_TOPOS, REWARD_SIGNIFICANT_DIGIT, REL_TOL
from oracle4grid.core.utils.constants import EnvConstants
from oracle4grid.core.utils.serialization import draw_graph, serialize_reward_df, serialize, display_topo_count, serialize_graph


def oracle(atomic_actions, env, debug, config, debug_directory=None,agent_seed=None,env_seed=None,
           grid_path=None, chronic_scenario=None, constants=EnvConstants()):
    """
    Compute best oracle path on scenario given set of topological unitary actions to consider and maximum combinatorial depth of those unitary actions

    Parameters
    ----------
    atomic_actions: :class:`dictionnary`
        dictionnary of possible atomic actions - keys: substations - values: grid2op action description
    env: :class:`grid2op.Environment`
        Represents the environment for the agent
    debug: :class:`bool`
        True if wants more logs
    config: :class:`dictionary`
        dictionary giving possible config for Oracle such as max_depth. See config.ini
    debug_directory: :class:`str`
        path where to save debug files for inspection
    agent_seed: :class:`int`
        agent seed on scenario when running grid2op
    env_seed :class:`int`
        env seed on scenario when running grid2op
    grid_path: :class:`str`
        path to the grid Grid2op environment
    chronic_scenario: :class:`str`
        name of scenario of interest


    Returns
    -------
    best_path, grid2op_action_path, best_path_no_overload, grid2op_action_path_no_overload, kpis
    grid2op_action_path: :class:`Grid2op Action`
        list of grid2op actions from best computation path
    best_path: :class:`Oracle Action`
        list of oracle actions from best computation path
    grid2op_action_path_no_overload: :class:`Grid2op Action`
        list of grid2op actions from computation path that has no overload at all (but possibly less optimal)
    best_path_no_overload: :class:`Oracle Action`
        list of oracle actions from best computation path that has no overload at all (but possibly less optimal)
    kpis: :class: 'Pandas Dataframe'
        dataframe of reward value of several baseline paths:  (donothing, best_ref_topo, best_path_no_overload,best_path,best_without_transitions)

    """
    # 0 - Preparation : Get initial topo and line status
    # init_topo_vect, init_line_status = get_initial_configuration(env)

    # 1 - Action generation step
    start_time = time.time()
    actions = combinator.generate(atomic_actions, int(config[MAX_DEPTH]), env, debug, nb_process=int(config[NB_PROCESS]))
    elapsed_time = time.time() - start_time
    print("elapsed_time for action generation is:"+str(elapsed_time))

    # 2 - Actions rewards simulation
    start_time = time.time()
    reward_df = run_many.run_all(actions, env, int(config[MAX_ITER]), int(config[NB_PROCESS]), debug=debug,
                                 agent_seed=agent_seed,env_seed=env_seed)
    elapsed_time = time.time() - start_time
    print("elapsed_time for simulation is:"+str(elapsed_time))

    if debug:
        print(reward_df)
        serialize_reward_df(reward_df, debug_directory)

    # 2.A Adding attacks node, a subgraph for each attack to allow topological actions within an action
    start_time = time.time()
    reward_df, windows = multiverse_simulation(env, actions, reward_df, debug,int(config[NB_PROCESS]), env_seed=env_seed, agent_seed=agent_seed)

    print("Windows of attack are :")
    for el in windows.items():
        print('attack_window: ' + el[0] + ' with lines possibly attacked ' + str([atk for atk in el[1]]))
    elapsed_time = time.time() - start_time
    print("elapsed_time for attack multiversing is:"+str(elapsed_time))

    # 3 - Graph generation
    start_time = time.time()
    if config[REWARD_SIGNIFICANT_DIGIT] is None:
        config[REWARD_SIGNIFICANT_DIGIT]='5' #already a large number of significant digit by default, and avoiding None problems

    graph = graph_generator.generate(reward_df, int(config[MAX_ITER])
                                     , debug=debug,reward_significant_digit=config[REWARD_SIGNIFICANT_DIGIT], constants=constants)
    elapsed_time = time.time() - start_time
    print("elapsed_time for graph creation is:"+str(elapsed_time))
    if debug:
        if len(graph.nodes)<100:
            serialize(graph, name="graphe", dir=debug_directory)
            draw_graph(graph, int(config[MAX_ITER]), save=debug_directory)
        serialize_graph(graph, debug_directory)



    # 4 - Best path computation (returns actions.npz + a list of atomic action dicts??)
    start_time = time.time()
    raw_path, best_path, grid2op_action_path = compute_trajectory.best_path(graph, config["best_path_type"], actions,
                                                                  debug=debug)
    raw_path_no_overload, best_path_no_overload, grid2op_action_path_no_overload = compute_trajectory.best_path_no_overload(graph, config["best_path_type"], actions,
                                                                  debug=debug)
    elapsed_time = time.time() - start_time
    print("elapsed_time for best_path computation is:"+str(elapsed_time))

    if debug:
        print("With possible overloads")
        print(raw_path)
        # Serialization for agent replay
        serialize(grid2op_action_path, 'best_path_grid2op_action',
                  dir=debug_directory, format='pickle')
        topo_count = display_topo_count(best_path, dir=debug_directory)
        print('10 best topologies in optimal path')
        print(topo_count)

        # Serialization for path with no overload
        if(len(best_path_no_overload)>=1):
            print("Without overload")
            print(raw_path_no_overload)
            serialize(grid2op_action_path_no_overload, 'best_path_grid2op_action_no_overload',
                      dir=debug_directory, format='pickle')
            topo_count = display_topo_count(best_path_no_overload, dir=debug_directory, name="best_path_no_overload_topologies_count.png")
            print('10 best topologies in optimal path')
            print(topo_count)

    # 5 - Indicators computation

    kpis = indicators.generate(raw_path, raw_path_no_overload, best_path, best_path_no_overload, reward_df, config["best_path_type"], int(config[N_TOPOS]),int(config['reward_significant_digit']), debug=debug)
    if debug:
        print(kpis)
        kpis.to_csv(os.path.join(debug_directory, "kpis.csv"), sep=';', index=False)

    # 6 - Replay of best path in real game rules condition
    replay_results = agent_replay.replay(grid2op_action_path, int(config[MAX_ITER]),
                                         kpis, grid_path, chronic_scenario, debug=debug, constants=constants,
                                         env_seed=env_seed, agent_seed=agent_seed, rel_tol=float(config[REL_TOL]), path_logs=debug_directory,oracle_action_path=best_path)

    if(len(grid2op_action_path_no_overload)>=1):
        replay_results_no_overload = agent_replay.replay(grid2op_action_path_no_overload, int(config[MAX_ITER]),
                                                         kpis, grid_path, chronic_scenario, debug=debug, constants=constants,
                                                         env_seed=env_seed, agent_seed=agent_seed,
                                                         rel_tol=float(config[REL_TOL]), path_logs=debug_directory,logs_file_name_extension="no_overload",oracle_action_path=best_path_no_overload)

    if debug:
        print("Number of survived timestep in replay: "+str(replay_results))
    return best_path, grid2op_action_path, best_path_no_overload, grid2op_action_path_no_overload, kpis




def save_oracle_data_for_replay(oracle_action_list,path_save):#(env,episode_name,grid2op_action_list,oracle_action_list,path_save):
    """
    Save oracle action list of best path by oracle action names in order to be easily reloaded after this heavy computation

    Parameters
    ----------
    scenario_folder: :class:`OracleAction`
        oracle action list of best path
    path_save: :class:`str`
        file path to save in

    """
    #nb_timesteps=len(oracle_action_list)
    #episode=init_episode_data(env,episode_name,nb_timesteps,path_save)
    #timestep=0
    #efficient_storing=True
    #for action in grid2op_action_list:
    #    episode.actions.update(timestep, action, efficient_storing)
    #    timestep+=1
#
    #episode.to_disk()

    # save oracle-action-path by names
    best_path_oracle_actions_names = [oracle_action.repr for oracle_action in oracle_action_list]
    pd.DataFrame(data=best_path_oracle_actions_names).to_csv(os.path.join(path_save, "oracle_actions.csv"), header=False,
                                                          index=False, sep=";")

#def init_episode_data(env,episode_name, nb_timestep_max,path_save):
#    disc_lines_templ = np.full(
#        (1, env.backend.n_line), fill_value=False, dtype=dt_bool)
#
#    attack_templ = np.full(
#        (1, env._oppSpace.action_space.size()), fill_value=0., dtype=dt_float)
#    times = np.full(nb_timestep_max, fill_value=np.NaN, dtype=dt_float)
#    rewards = np.full(nb_timestep_max, fill_value=np.NaN, dtype=dt_float)
#    actions = np.full((nb_timestep_max, env.action_space.n),
#                      fill_value=np.NaN, dtype=dt_float)
#    env_actions = np.full(
#        (nb_timestep_max, env._helper_action_env.n), fill_value=np.NaN, dtype=dt_float)
#    observations = np.full(
#        (nb_timestep_max + 1, env.observation_space.n), fill_value=np.NaN, dtype=dt_float)
#    disc_lines = np.full(
#        (nb_timestep_max, env.backend.n_line), fill_value=np.NaN, dtype=dt_bool)
#    attack = np.full((nb_timestep_max, env._opponent_action_space.n), fill_value=0., dtype=dt_float)
#
#    import logging
#    logger = logging.getLogger(__name__)
#    episode = EpisodeData(actions=actions,
#                          env_actions=env_actions,
#                          observations=observations,
#                          rewards=rewards,
#                          disc_lines=disc_lines,
#                          times=times,
#                          observation_space=env.observation_space,
#                          action_space=env.action_space,
#                          helper_action_env=env._helper_action_env,
#                          path_save=path_save,
#                          disc_lines_templ=disc_lines_templ,
#                          attack_templ=attack_templ,
#                          attack=attack,
#                          attack_space=env._opponent_action_space,
#                          logger=logger,
#                          name=episode_name,#env.chronics_handler.get_name(),
#                          force_detail=True,
#                          other_rewards=[])
#    episode.set_parameters(env)
#    return episode

def load_and_run(env_dir, chronic, action_file, debug,agent_seed,env_seed, config, constants=EnvConstants()):
    atomic_actions, env, debug_directory, chronic_id = load(env_dir, chronic, action_file, debug, constants=constants, config = config)
    # Parse atomic_actions format
    # atomic_actions = parse(atomic_actions,env)
    parser = OracleParser(atomic_actions, env.action_space)
    atomic_actions = parser.parse()

    # Run all steps
    return oracle(atomic_actions, env, debug, config, debug_directory=debug_directory,agent_seed=agent_seed,env_seed=env_seed,
                  grid_path=env_dir, chronic_scenario=chronic, constants=constants)

def load_oracle_data_for_replay(env,action_file,path_save,action_depth=1,nb_process=1):
    """
    Save oracle action list of best path by oracle action names

    Parameters
    ----------
    scenario_folder: :class:`OracleAction`
        oracle action list of best path
    action_file: :class:`str`
        file path where possible unitary actions are described
    path_save: :class:`str`
        file path to save in
    action_depth: :class:`int`
        maximum possible combinatorial depth of unitary actions
    nb_process: :class:`int`
        number of cores to generate all possible comnbined actions in parallel


    Returns
    -------
    action_list_reloaded: :class:`Grid2op Action`
        list of grid2op actions reloaded from best computation path
    oracle_actions_in_path: :class:`Oracle Action`
        list of oracle actions reloaded from best computation path
    init_topo_vect: class:`numpy.ndarray`, dtype:int
        topo vect at start time
    init_line_status: class:`numpy.ndarray`, dtype:bool
        line status at start time
    """
    #episode_reload=EpisodeData.from_disk(path_save,episode_name)
    #action_list_reloaded=episode_reload.actions.objects
    init_topo_vect, init_line_status = get_initial_configuration(env)

    with open(action_file) as f:
        atomic_actions = json.load(f)

    parser = OracleParser(atomic_actions, env.action_space)
    atomic_actions = parser.parse()

    oracle_actions = combinator.generate(atomic_actions, action_depth, env, debug=False,
                                  nb_process=nb_process)

    oracle_actions_map = dict({oracle_action.repr: oracle_action for oracle_action in oracle_actions})

    oracle_actions_name_in_path = pd.read_csv(os.path.join(path_save, "oracle_actions.csv"), header=None)
    oracle_actions_in_path=[oracle_actions_map[name_oracle_action] for name_oracle_action in oracle_actions_name_in_path[0]]

    action_list_reloaded=[oracle_actions_in_path[i].grid2op_action for i in range(len(oracle_actions_in_path))]

    return action_list_reloaded, init_topo_vect, init_line_status, oracle_actions_in_path
