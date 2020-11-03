import os
from datetime import datetime
import pickle

from oracle4grid.core.graph import graph_generator, compute_trajectory
from oracle4grid.core.utils.prepare_environment import get_initial_configuration
from oracle4grid.core.reward_computation import run_many
from oracle4grid.core.actions_utils import combinator
from oracle4grid.core.utils.config_ini_utils import MAX_ITER, MAX_DEPTH, NB_PROCESS


# runs all steps one by one
# handles visualisation in each step

def oracle(atomic_actions, env, debug, config, debug_directory=None):
    # 0 - Preparation
    # Get initial topo and line status
    init_topo_vect, init_line_status = get_initial_configuration(env)

    # 1 - Action generation step
    actions = combinator.generate(atomic_actions, int(config[MAX_DEPTH]), env, debug, init_topo_vect, init_line_status)
    if debug:
        print(actions)

    # 2 - Actions rewards simulation
    reward_df = run_many.run_all(actions, env, int(config[MAX_ITER]), int(config[NB_PROCESS]))
    if debug:
        print(reward_df)
        serialize(reward_df, name='reward_df', dir=debug_directory)
        # TODO: ligne de commande qui permet de charger à partir d'ici

    # 3 - Graph generation
    # TODO: traiter les actions qui ne sont pas allées au bout des timesteps
    graph = graph_generator.generate(reward_df, int(config['max_depth']), init_topo_vect, init_line_status)
    if debug:
        print(graph)
        serialize(graph, name="graphe", dir=debug_directory)

    # 4 - Best path computation (returns actions.npz + a list of atomic action dicts??)
    best_path = compute_trajectory.best_path(graph, config["best_path_type"])
    if debug:
        print(best_path)

    # 5 - Indicator computations

    return reward_df


def serialize(obj, name, dir):
    outfile = open(os.path.join(dir, name + ".pkl"), 'wb')
    pickle.dump(obj, outfile)
    outfile.close()
