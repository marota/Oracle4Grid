import operator
from pprint import pprint
from itertools import combinations
from functools import reduce
from tqdm import tqdm

from oracle4grid.core.utils.actions_generator import get_atomic_actions_names
from oracle4grid.core.utils.Action import OracleAction
# from oracle4grid.core.reward_computation.run_one import run_one

# should output a dict with all combinations of actions based on dict of possible actions


def generate(atomic_actions, depth, env, debug):
    ret = []
    all_actions = generate_all(atomic_actions, depth, env)
    if debug:
        print("Initial atomic actions")
        pprint(atomic_actions)
        print('\nExample of stored combination in Action class')
        all_actions[-1].print()
    for action in all_actions:
        env.reset()
        if keep(action, env, debug):
            ret.append(action)
    if debug:
        print(str(len(all_actions)-len(ret))+" actions out of "+str(len(all_actions))+" have been filtered")
    return ret


def generate_all(atomic_actions, depth, env):
    # don't generate associative actions
    named_atomic_actions = get_atomic_actions_names(atomic_actions)
    all_names_combinations = reduce(operator.concat, [list(combinations(named_atomic_actions, i)) for i in range(1, (depth + 1))])
    init_topo_vect = env.get_obs().topo_vect
    all_actions = [OracleAction(names_combination, [named_atomic_actions[name] for name in names_combination],
                                env.action_space, init_topo_vect)
                   for names_combination in all_names_combinations]
    return all_actions


def keep(oracle_action, env, debug=False):
    # Successive rules applied to invalidate actions that don't need to be simulated
    check = run_pf_check(oracle_action, env, debug)
    return True


def run_pf_check(oracle_action, env, debug=False):
    valid = False

    # First step simulation with Runner
    # run = run_one(oracle_action, env, max_iter=1)
    obs, reward, done, info = env.step(oracle_action.grid2op_action)

    # Valid action?
    if (not done) and (len(info["exception"]) == 0):
        valid = True

    if debug:
        print(oracle_action.atomic_actions)
        print(info)
        print('\n')
    return valid
