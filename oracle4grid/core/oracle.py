from oracle4grid.core.actions_utils import combinator

#runs all steps one by one
#handles visualisation in each step

def oracle(atomic_actions, env, debug, config):
    # 1 - Action generation step
    actions = combinator.generate(atomic_actions, config["maxDepth"], env, debug)

    # 2 - Actions rewards simulation
    # 3 - Graph generation
    # 4 - Best path computation (returns actions.npz + a list of atomic action dicts??)
    # 5 - Indicator computations
    return 1 # a list of atomic action dicts?