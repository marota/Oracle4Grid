import os
import time

from oracle4grid.core.graph import graph_generator, compute_trajectory, indicators
from oracle4grid.core.utils.prepare_environment import get_initial_configuration
from oracle4grid.core.reward_computation import run_many
from oracle4grid.core.actions_utils import combinator
from oracle4grid.core.utils.config_ini_utils import MAX_ITER, MAX_DEPTH, NB_PROCESS, N_TOPOS
from oracle4grid.core.utils.serialization import draw_graph, serialize_reward_df, serialize, display_topo_count,serialize_graph


def oracle(atomic_actions, env, debug, config, debug_directory=None,agent_seed=None,env_seed=None):
    # 0 - Preparation : Get initial topo and line status
    init_topo_vect, init_line_status = get_initial_configuration(env)

    # 1 - Action generation step
    start_time = time.time()
    actions = combinator.generate(atomic_actions, int(config[MAX_DEPTH]), env, debug, init_topo_vect, init_line_status)
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
        # serialize(reward_df, name='reward_df', dir=debug_directory)
        # reward_df = load_serialized_object('reward_df', debug_directory)

    # 3 - Graph generation
    start_time = time.time()
    graph = graph_generator.generate(reward_df, init_topo_vect, init_line_status, int(config[MAX_ITER]), debug=debug)
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
                                                                  init_topo_vect, init_line_status, debug=debug)
    elapsed_time = time.time() - start_time
    print("elapsed_time for best_path computation is:"+str(elapsed_time))

    if debug:
        print(best_path)
        # Serialization for agent replay
        serialize(grid2op_action_path, 'best_path_grid2op_action',
                  dir=debug_directory, format='pickle')
        topo_count = display_topo_count(best_path, dir=debug_directory)
        print('10 best topologies in optimal path')
        print(topo_count)

    # 5 - Indicators computation
    kpis = indicators.generate(best_path, reward_df, config["best_path_type"], int(config[N_TOPOS]), debug=debug)
    if debug:
        print(kpis)
        kpis.to_csv(os.path.join(debug_directory, "kpis.csv"), sep=';', index=False)

    return best_path, grid2op_action_path, kpis
