from oracle4grid.core.reward_computation import run_many
from oracle4grid.core.actions_utils import combinator

#runs all steps one by one
#handles visualisation in each step

def oracle(atomic_actions, env, debug, config):
    # 1 - Action generation step
    actions = combinator.generate(atomic_actions, config["maxDepth"], env, debug)
    if debug:
        print(actions)
    # 2 - Actions rewards simulation
    reward_df = run_many.run_all(actions, config, env)
    if debug:
        print(reward_df)
    # 3 - Graph generation
    # 4 - Best path computation (returns actions.npz + a list of atomic action dicts??)
    # 5 - Indicator computations

    return reward_df

