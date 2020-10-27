from oracle4grid.core.graph import graph_generator, compute_trajectory
from oracle4grid.core.reward_computation import run_many
from oracle4grid.core.actions_utils import combinator
from oracle4grid.core.utils.config_ini_utils import MAX_ITER, MAX_DEPTH, NB_PROCESS


# runs all steps one by one
# handles visualisation in each step

def oracle(atomic_actions, env, debug, config):
    # 1 - Action generation step
    actions = combinator.generate(atomic_actions, int(config[MAX_DEPTH]), env, debug)
    if debug:
        print(actions)
    # 2 - Actions rewards simulation
    reward_df = run_many.run_all(actions, env, int(config[MAX_ITER]), int(config[NB_PROCESS]))
    if debug:
        print(reward_df)
    # 3 - Graph generation
    graph = graph_generator.generate(reward_df, config, env, actions)
    if debug:
        print(graph)
    # 4 - Best path computation (returns actions.npz + a list of atomic action dicts??)
    best_path = compute_trajectory.best_path(graph, config)
    if debug:
        print(best_path)
    # 5 - Indicator computations

    return reward_df
