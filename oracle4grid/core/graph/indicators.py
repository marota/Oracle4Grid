import pandas as pd

SHORTEST = "shortest"
LONGEST = "longest"
MSG_DONOTHING = "Do Nothing in ref topo"
MSG_TOPO = " then do nothing"
MSG_GAMERULES = "Best possible path with game rules"
MSG_NOGAMERULE = "Best path without transition constraint"

def generate(best_path, reward_df, best_path_type, N, debug = False):
    if debug:
        print('\n')
        print("============== 5 - Indicators computation ==============")

    reward_df['name'] = [action.name for action in reward_df['action']]
    results = []

    # Pure donothing
    donothing = compute_reward_donothing(reward_df)
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

    indicators = pd.DataFrame(data=results,
                              columns=['Indicator name', 'Reward value']).sort_values(by = 'Reward value',
                                                                                      ascending=(best_path_type==SHORTEST))
    # Check if systematic order is respected
    check, message = check_indicators_order(donothing[1], best[1], best_without_transitions[1], best_path_type)
    if not check:
        raise ValueError(message)
    return indicators

def check_indicators_order(donothing, best, best_without_transitions, best_path_type):
    check = True
    message = ""

    if best_path_type == SHORTEST:
        # Check if best path better than donothing
        if donothing < best:
            check = False
            message = "Indicator '"+MSG_DONOTHING+"' is better than indicator '"+MSG_GAMERULES
        # Check if best without game rule better than best with game rules
        if best < best_without_transitions:
            check = False
            message = "Indicator '"+MSG_GAMERULES+"' is better than indicator '"+MSG_NOGAMERULE
    elif best_path_type == LONGEST:
        # Check if best path better than donothing
        if donothing > best:
            check = False
            message = "Indicator '"+MSG_DONOTHING+"' is better than indicator '"+MSG_GAMERULES+ " which should not be possible"
        # Check if best without game rule better than best with game rules
        if best > best_without_transitions:
            check = False
            message = "Indicator '"+MSG_GAMERULES+"' is better than indicator '"+MSG_NOGAMERULE+ " which should not be possible"
    return check, message

def get_best_path_reward(best_path, reward_df):
    reward = 0.
    for t ,action in enumerate(best_path):
        reward += reward_df.loc[(reward_df['name']==action.name)&(reward_df['timestep']==t),'reward'].values[0]
    return [MSG_GAMERULES, reward]

def get_best_path_without_constraints(reward_df, best_path_type):
    if best_path_type == SHORTEST:
        fun = 'min'
    elif best_path_type== LONGEST:
        fun = 'max'
    else:
        raise ValueError('best path type in config.ini should be longest or shortest')
    reward = reward_df.groupby('timestep').agg({'reward':fun})['reward'].sum()
    return [MSG_NOGAMERULE, reward]

def compute_reward_topo_then_donothing(reward_df, topos):
    results = []
    for topo in topos:
        reward = reward_df.loc[reward_df['name']==topo.name,'reward'].sum()
        results.append([str(topo)+MSG_TOPO, reward])
    return results

def get_best_topos(best_path, N):
    path = pd.Series(best_path)
    n_unique_topos = path.nunique()
    if n_unique_topos > N:
        N = n_unique_topos
    best_topos_series = path.value_counts().head(N)
    return best_topos_series.index.tolist()


def compute_reward_donothing(reward_df, donothing_id = 0):
    donothing = reward_df.loc[reward_df['name']==donothing_id,'reward'].sum()
    return [MSG_DONOTHING, donothing]

