import operator
from pprint import pprint
from itertools import combinations
from functools import reduce

from oracle4grid.core.utils.actions_generator import get_atomic_actions_names
from oracle4grid.core.utils.Action import OracleAction



def generate(atomic_actions, depth, env, debug, init_topo_vect, init_line_status):
    if debug:
        print('\n')
        print("============== 1 - Generation of action combinations ==============")
    ret = []
    all_actions = generate_all(atomic_actions, depth, env, init_topo_vect, init_line_status)
    for action in all_actions:
        env.reset()
        if keep(action, env):
            ret.append(action)
    if debug:
        print(str(len(all_actions)-len(ret))+" actions out of "+str(len(all_actions))+" have been filtered")
    return ret


def generate_all(atomic_actions, depth, env, init_topo_vect, init_line_status):
    # don't generate associative actions
    named_atomic_actions = get_atomic_actions_names(atomic_actions)
    all_names_combinations = reduce(operator.concat, [list(combinations(named_atomic_actions, i)) for i in range(1, (depth + 1))])

    all_actions = [OracleAction(i, names_combination, [named_atomic_actions[name] for name in names_combination],
                                env.action_space,
                                init_topo_vect,
                                init_line_status)
                   for i, names_combination in enumerate(all_names_combinations)]
    return all_actions


def keep(oracle_action, env):
    # Successive rules applied to invalidate actions that don't need to be simulated
    check = run_pf_check(oracle_action, env)
    return True


def run_pf_check(oracle_action, env):
    valid = False

    # First step simulation with Runner
    # run = run_one(oracle_action, env, max_iter=1)
    obs, reward, done, info = env.step(oracle_action.grid2op_action)

    # Valid action?
    if (not done) and (len(info["exception"]) == 0):
        valid = True
    return valid
