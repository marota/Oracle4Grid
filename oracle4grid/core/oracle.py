from oracle4grid.core.actions_utils import combinator

#runs all steps one by one
#handles visualisation in each step

def oracle(atomic_actions, env, debug, config):
    actions = combinator.generate(atomic_actions, config["maxDepth"], env, debug)
    return 1 #dataframe of all results and chosen path