import pandas as pd


SHORTEST = "shortest"
LONGEST = "longest"
MSG_DONOTHING = "Do Nothing in ref topo"
MSG_TOPO = " then do nothing"
MSG_GAMERULES_NO_OL = "Best possible path with game rules and no overload"
MSG_GAMERULES = "Best possible path with game rules"
MSG_NOGAMERULE = "Best path without transition constraint"


def generate(raw_path, raw_path_no_overload, best_path, best_path_no_overload, reward_df, best_path_type, N, debug=False):
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

    # Best path without overload
    best_no_ol = get_best_path_reward_no_overload(raw_path_no_overload, reward_df)
    results.append(best_no_ol)

    # Best path
    best = get_best_path_reward(raw_path, reward_df)
    results.append(best)

    # Best path without taking care of possible transition
    best_without_transitions = get_best_path_without_constraints(reward_df, best_path_type)
    results.append(best_without_transitions)

    indicators = pd.DataFrame(data=results,
                              columns=['Indicator name', 'Reward value']).sort_values(by='Reward value',
                                                                                      ascending=(best_path_type == SHORTEST))
    # Check if systematic order is respected
    check, message = check_indicators_order(donothing[1], best[1], best_no_ol[1], best_without_transitions[1], best_path_type)
    if not check:
        raise ValueError(message)
    return indicators


def check_indicators_order(donothing, best, best_no_ol, best_without_transitions, best_path_type):
    check = True
    message = ""

    if best_path_type == SHORTEST:
        # Check if best path better than donothing
        if donothing < best:
            check = False
            message = "Indicator '" + MSG_DONOTHING + "' is better than indicator '" + MSG_GAMERULES
        # Check if best without game rule better than best with game rules
        if best < best_without_transitions:
            check = False
            message = "Indicator '" + MSG_GAMERULES + "' is better than indicator '" + MSG_NOGAMERULE
        # Check if best path better than best without overload
        if pd.notnull(best_no_ol) and best_no_ol < best:
            check = False
            message = "Indicator '" + MSG_GAMERULES_NO_OL + "' is better than indicator '" + MSG_GAMERULES
    elif best_path_type == LONGEST:
        # Check if best path better than donothing
        if donothing > best:
            check = False
            message = "Indicator '" + MSG_DONOTHING + "' is better than indicator '" + MSG_GAMERULES + " which should not be possible"
        # Check if best without game rule better than best with game rules
        if best > best_without_transitions:
            check = False
            message = "Indicator '" + MSG_GAMERULES + "' is better than indicator '" + MSG_NOGAMERULE + " which should not be possible"
        # Check if best path better than best without overload
        if pd.notnull(best_no_ol) and best_no_ol > best:
            check = False
            message = "Indicator '" + MSG_GAMERULES_NO_OL + "' is better than indicator '" + MSG_GAMERULES
    return check, message


def get_best_path_reward(best_path, reward_df):
    reward = 0.
    for t, action in enumerate(best_path):
        reward += reward_df.loc[reward_df['node_name'] == action, 'reward'].values[0]
    return [MSG_GAMERULES, reward]


def get_best_path_reward_no_overload(best_path_no_overload, reward_df):
    if len(best_path_no_overload) == 0:
        reward = float('nan')
    else:
        reward = 0.
        for t, action in enumerate(best_path_no_overload):
            reward += reward_df.loc[(reward_df['node_name'] == action), 'reward'].values[0]
    return [MSG_GAMERULES_NO_OL, reward]


def get_best_path_without_constraints(reward_df, best_path_type):
    if best_path_type == SHORTEST:
        fun = 'min'
    elif best_path_type == LONGEST:
        fun = 'max'
    else:
        raise ValueError('best path type in config.ini should be longest or shortest')
    reward = reward_df.groupby('timestep').agg({'reward': fun})['reward'].sum()
    return [MSG_NOGAMERULE, reward]


def compute_reward_topo_then_donothing(reward_df, topos):
    results = []
    for topo in topos:
        reward = get_reward_from_topo_id(reward_df, topo.name)
        results.append([str(topo) + MSG_TOPO, reward])
    return results


def get_best_topos(best_path, N):
    path = pd.Series(best_path)
    n_unique_topos = path.nunique()
    if n_unique_topos > N:
        N = n_unique_topos
    best_topos_series = path.value_counts().head(N)
    return best_topos_series.index.tolist()


def compute_reward_donothing(reward_df, donothing_id=0):
    donothing = get_reward_from_topo_id(reward_df, donothing_id)
    return [MSG_DONOTHING, donothing]


def get_reward_from_topo_id(reward_df, id):
    return reward_df[reward_df['multiverse'] == False].loc[reward_df['name'] == id, 'reward'].sum()
