import os
import time

from oracle4grid.core.graph import graph_generator, compute_trajectory, indicators
from oracle4grid.core.replay import agent_replay
from oracle4grid.core.utils.constants import EnvConstants
from oracle4grid.core.utils.prepare_environment import get_initial_configuration
from oracle4grid.core.reward_computation import run_many
from oracle4grid.core.actions_utils import combinator
from oracle4grid.core.utils.config_ini_utils import MAX_ITER, MAX_DEPTH, NB_PROCESS, N_TOPOS,REWARD_SIGNIFICANT_DIGIT
from oracle4grid.core.utils.serialization import draw_graph, serialize_reward_df, serialize, display_topo_count,serialize_graph


def oracle(atomic_actions, env, debug, config, debug_directory=None,agent_seed=None,env_seed=None,
           reward_significant_digit=None, grid_path=None, chronic_id=None, constants=EnvConstants()):
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
    best_path, grid2op_action_path = compute_trajectory.best_path(graph, config["best_path_type"], actions,
                                                                  debug=debug)
    best_path_no_overload, grid2op_action_path_no_overload = compute_trajectory.best_path_no_overload(graph, config["best_path_type"], actions,
                                                                  debug=debug)
    elapsed_time = time.time() - start_time
    print("elapsed_time for best_path computation is:"+str(elapsed_time))

    if debug:
        print("With possible overloads")
        print(best_path)
        # Serialization for agent replay
        serialize(grid2op_action_path, 'best_path_grid2op_action',
                  dir=debug_directory, format='pickle')
        topo_count = display_topo_count(best_path, dir=debug_directory)
        print('10 best topologies in optimal path')
        print(topo_count)

        # Serialization for path with no overload
        print("Without overload")
        print(best_path_no_overload)
        serialize(grid2op_action_path_no_overload, 'best_path_grid2op_action_no_overload',
                  dir=debug_directory, format='pickle')
        topo_count = display_topo_count(best_path_no_overload, dir=debug_directory, name="best_path_no_overload_topologies_count.png")
        print('10 best topologies in optimal path')
        print(topo_count)

    # 5 - Indicators computation
    kpis = indicators.generate(best_path, reward_df, config["best_path_type"], int(config[N_TOPOS]), debug=debug)
    if debug:
        print(kpis)
        kpis.to_csv(os.path.join(debug_directory, "kpis.csv"), sep=';', index=False)

    # 6 - Replay of best path in real game rules condition
    replay_results = agent_replay.replay(grid2op_action_path, int(config[MAX_ITER]),
                                         kpis, grid_path, chronic_id, debug=debug, constants=constants,
                                         env_seed=env_seed, agent_seed=agent_seed)
    if debug:
        print("Number of survived timestep in replay: "+str(replay_results))
    return best_path, grid2op_action_path, best_path_no_overload, grid2op_action_path_no_overload, kpis
