import pandas as pd

SHORTEST = "shortest"
LONGEST = "longest"

def generate(best_path, reward_df, best_path_type, N, debug = False):
    if debug:
        print('\n')
        print("============== 5 - Indicators computation ==============")

    reward_df['name'] = [action.name for action in reward_df['action']]
    results = []

    # Pure donothing
    # TODO: simulation?
    donothing = compute_reward_donothing()
    results.append(donothing)

    # Apply best topos and then do nothing
    topos = get_best_topos(best_path, N)
    topo_then_donothing = compute_reward_topo_then_donothing(reward_df, topos)
    results += topo_then_donothing

    # Best path
    best = get_best_path_reward(best_path, reward_df)
    results.append(best)

    # Best path without taking care of possible transition
    best_without_transitions = get_best_path_without_constraints(reward_df, best_path_type)
    results.append(best_without_transitions)

    # Best path with only one action transitions allowed
    # TODO: generer autre graphe? OU ELAGUER LE GRAPH ACTUEL? PASSE EN ARGUMENT
    best_with_oneaction_transitions = get_best_path_with_oneaction_transitions(reward_df, best_path_type)
    results.append(best_with_oneaction_transitions)

    indicators = pd.DataFrame(data=results,
                              columns=['Indicator name', 'Reward value']).sort_values(by = 'Reward value',
                                                                                      ascending=(best_path_type==SHORTEST))
    return indicators

def get_best_path_reward(best_path, reward_df):
    reward = 0.
    for t ,action in enumerate(best_path):
        reward += reward_df.loc[(reward_df['name']==action.name)&(reward_df['timestep']==t),'reward'].values[0]
    return ['Best possible path with game rules', reward]

def get_best_path_without_constraints(reward_df, best_path_type):
    if best_path_type == SHORTEST:
        fun = 'min'
    elif best_path_type== LONGEST:
        fun = 'max'
    else:
        raise ValueError('best path type in config.ini should be longest or shortest')
    reward = reward_df.groupby('timestep').agg({'reward':fun})['reward'].sum()
    return ['Best path without transition constraint', reward]

def get_best_path_with_oneaction_transitions(reward_df, best_path_type):
    return ['Best path with only one-action transitions', 0.]


def compute_reward_topo_then_donothing(reward_df, topos):
    results = []
    for topo in topos:
        reward = reward_df.loc[reward_df['name']==topo.name,'reward'].sum()
        results.append(["Apply "+str(topo)+" then do nothing", reward])
    return results

def get_best_topos(best_path, N):
    path = pd.Series(best_path)
    n_unique_topos = path.nunique()
    if n_unique_topos > N:
        N = n_unique_topos
    best_topos_series = path.value_counts().head(N)
    return best_topos_series.index.tolist()


def compute_reward_donothing():
    # TODO
    return ["Do Nothing in ref topo", 0.]

