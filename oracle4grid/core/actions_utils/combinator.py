import operator
from pprint import pprint
from itertools import combinations
from functools import reduce
from tqdm import tqdm

from oracle4grid.core.utils.actions_generator import get_atomic_actions_names
from oracle4grid.core.utils.Action import Action
#should output a dict with all combinations of actions based on dict of possible actions


def generate(atomic_actions, depth, env, debug) :
    ret = []
    all_actions = generate_all(atomic_actions, depth)
    for action in all_actions :
        if keep(action, env):
            ret.append(action)
    print(ret)
    return ret

def generate_all(atomic_actions, depth) :
    # don't generate associative actions
    depth = int(depth)
    named_atomic_actions = get_atomic_actions_names(atomic_actions)
    all_names_combinations = reduce(operator.concat, [list(combinations(named_atomic_actions,i)) for i in range(1, (depth + 1))])
    all_actions = [Action([named_atomic_actions[name] for name in names_combination]) for names_combination in all_names_combinations]
    return all_actions


def keep(action, env) :
    check = run_pf_check(action, env)
    return check

def run_pf_check(action,env) :
    return True



