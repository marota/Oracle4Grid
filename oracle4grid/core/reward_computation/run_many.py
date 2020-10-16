# outputs a dataframe with rewards for each run (run = N actions) and each timestep
# takes a dict of all combinations of actions


def run_all(actions, ini, env):
    df = parallel(env, actions, ini)
    return df


def parallel(env, actions, ini):
    # run in parallel all actions and return df of rewards indexed by timestep, atomic actions and subs
    # ["time", "Action", "reward"]
