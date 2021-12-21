import os
import time

from oracle4grid.core.actions_utils import combinator
from oracle4grid.core.graph import graph_generator, compute_trajectory, indicators
from oracle4grid.core.replay import agent_replay
from oracle4grid.core.reward_computation import run_many
from oracle4grid.core.reward_computation.attacks_multiverse import multiverse_simulation
from oracle4grid.core.utils.config_ini_utils import MAX_ITER, MAX_DEPTH, NB_PROCESS, N_TOPOS, REWARD_SIGNIFICANT_DIGIT, REL_TOL
from oracle4grid.core.utils.constants import EnvConstants
from oracle4grid.core.utils.serialization import draw_graph, serialize_reward_df, serialize, display_topo_count, serialize_graph


def oracle(atomic_actions, env, debug, config, debug_directory=None,agent_seed=None,env_seed=None,
           reward_significant_digit=None, grid_path=None, chronic_scenario=None, constants=EnvConstants()):
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
    reward_df, windows = multiverse_simulation(env, actions, reward_df, debug, env_seed=env_seed, agent_seed=agent_seed)

    print("Windows of attack are :")
    for el in windows.items():
        print('attack_window: ' + el[0] + ' with lines possibly attacked ' + str([atk for atk in el[1]]))
    elapsed_time = time.time() - start_time
    print("elapsed_time for attack multiversing is:"+str(elapsed_time))

    # 3 - Graph generation
    start_time = time.time()
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
    kpis = indicators.generate(raw_path, raw_path_no_overload, best_path, best_path_no_overload, reward_df, config["best_path_type"], int(config[N_TOPOS]), debug=debug)
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
