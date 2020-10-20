import operator
from pprint import pprint
from itertools import combinations
from functools import reduce
from tqdm import tqdm

from oracle4grid.core.utils.actions_generator import get_atomic_actions_names
from oracle4grid.core.utils.Action import OracleAction
#should output a dict with all combinations of actions based on dict of possible actions


def generate(atomic_actions, depth, env, debug) :
    ret = []
    all_actions = generate_all(atomic_actions, depth, env)
    if debug:
        print("Initial atomic actions")
        pprint(atomic_actions)
        print('\nExample of stored combination in Action class')
        pprint(all_actions[-2].atomic_actions)
    for action in all_actions :
        if keep(action, env, debug):
            ret.append(action)
    print(ret)
    return ret

def generate_all(atomic_actions, depth, env) :
    # don't generate associative actions
    depth = int(depth)
    named_atomic_actions = get_atomic_actions_names(atomic_actions)
    all_names_combinations = reduce(operator.concat, [list(combinations(named_atomic_actions,i)) for i in range(1, (depth + 1))])
    init_topo_vect = [0,0,0,0,0]
    # TODO: fonction qui extrait ce topo vect complet initial, qu'il soit en statique
    all_actions = [OracleAction([named_atomic_actions[name] for name in names_combination], env.action_space, init_topo_vect) for names_combination in all_names_combinations]
    return all_actions


def keep(action, env, debug = False) :
    # Successive rules applied to invalidate actions that don't need to be simulated
    #check = run_pf_check(action.grid2op_action_dict, env, debug)
    return True

def run_pf_check(grid2op_action_dict,env, debug = False) :
    valid = False

    # First step simulation
    observation, reward, done, info = env.step(env.action_space({}))
    #         obs, reward, done, info = observation.simulate(env.action_space(s), time_step=0)#env.step(env.action_space(s))
    obs, reward, done, info = observation.simulate(env.action_space(grid2op_action_dict), time_step=0)
    env.reset() # For next action !

    # Valid action?
    if (not done) and (len(info["exception"]) == 0):
        valid = True

    if debug:
        print(action.atomic_actions)
        print(info)
        print('\n')
    return valid



