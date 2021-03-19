import multiprocessing
import operator
from pprint import pprint
from itertools import combinations, compress
from itertools import product
from functools import reduce
import numpy as np

from oracle4grid.core.utils.actions_generator import get_atomic_actions_names
from oracle4grid.core.utils.Action import OracleAction


def generate(atomic_actions, depth, env, debug, nb_process=1):
    if debug:
        print('\n')
        print("============== 1 - Generation of action combinations ==============")
    all_actions = generate_all(atomic_actions, depth, env)
    ret = filter_actions(all_actions, env, debug, nb_process)
    return ret


def generate_all(atomic_actions, depth, env):
    # don't generate associative actions
    named_atomic_actions, atomic_action_asset_dic = get_atomic_actions_names(atomic_actions)
    all_asset_combinations = reduce(operator.concat, [list(combinations(atomic_action_asset_dic.keys(), i))
                                                      for i in range(1, (depth + 1))])
    all_names_combinations = reduce(operator.concat, [list(product(*[atomic_action_asset_dic[asset] for asset in all_asset_combinations[i]]))
                                                      for i in range(len(all_asset_combinations))])

    all_actions = [OracleAction(i + 1, names_combination, [named_atomic_actions[name] for name in names_combination],
                                env.action_space)
                   for i, names_combination in enumerate(all_names_combinations)]
    # Add donothing action
    all_actions.append(OracleAction(0, ['donothing-0'], [], env.action_space))
    return all_actions


def filter_actions(all_actions, env, debug, nb_process):
    ret = []
    obs = env.reset()
    for action in all_actions:
        if keep_action(action, obs):
            ret.append(action)
    if debug:
        print(str(len(all_actions) - len(ret)) + " actions out of " + str(len(all_actions)) + " have been filtered")
    return ret


def keep_action(oracle_action, obs):
    # Successive rules applied to invalidate actions that don't need to be simulated
    if str(oracle_action) == "donothing-0":
        check_run = True
    else:
        check_run = run_pf_check(oracle_action, obs)
    return check_run


def run_pf_check(oracle_action, obs):
    init_topo_vect = obs.topo_vect
    init_line_status = obs.line_status

    valid = False
    sim_obs, sim_reward, sim_done, sim_info = obs.simulate(oracle_action.grid2op_action, time_step=0)

    # Valid action?
    if (not sim_done) and (len(sim_info["exception"]) == 0):
        valid = True

    # Check if has impact on subs
    new_topo_vect = sim_obs.topo_vect
    impact_on_subs = np.where(new_topo_vect!=init_topo_vect)[0]
    has_impact_on_subs = (len(impact_on_subs)!=0)

    # Check if has impact on lines
    new_line_status = sim_obs.line_status
    impact_on_lines = np.where(new_line_status != init_line_status)[0]
    has_impact_on_lines = (len(impact_on_lines) != 0)
    return (valid and (has_impact_on_subs or has_impact_on_lines))

