import json
import os

from oracle4grid.core.utils.prepare_environment import prepare_simulation_params, prepare_env
from oracle4grid.core.oracle import oracle


def load_and_run(env_dir, chronic, action_file, debug,agent_seed,env_seed, config):
    atomic_actions, env, debug_directory = load(env_dir, chronic, action_file, debug)
    # Parse atomic_actions format
    atomic_actions = parse(atomic_actions,env)
    # Run all steps
    return oracle(atomic_actions, env, debug, config, debug_directory=debug_directory,agent_seed=agent_seed,env_seed=env_seed)


def load(env_dir, chronic, action_file, debug):
    param = prepare_simulation_params()  # Move to ini?
    env = prepare_env(env_dir, chronic, param)

    # Load unitary actions
    with open(action_file) as f:
        atomic_actions = json.load(f)

    # Init debug mode if necessary
    if debug:
        debug_directory = init_debug_directory(env_dir, action_file, chronic)
    else:
        debug_directory = None
    return atomic_actions, env, debug_directory


def init_debug_directory(env_dir, action_file, chronic):
    action_file_os = os.path.split(action_file)[len(os.path.split(action_file)) - 1].replace(".json", "")
    grid_file_os = os.path.split(env_dir)[len(os.path.split(env_dir)) - 1]
    scenario = "scenario_" + str(chronic)
    debug_directory = os.path.join("oracle4grid/output/", grid_file_os, scenario, action_file_os)
    os.makedirs(debug_directory, exist_ok=True)
    return debug_directory

def parse(d, env):
    if type(d) is list:
        if 'set_bus' in list(d[0].keys()):
            if 'substations_id' in list(d[0]['set_bus'].keys()):
                # Format 1 detected
                print("Specific format is detected for actions: converting with parser")
                d = parser1(d,env)
                return d
    if type(d) is dict:
        if 'sub' in list(d.keys()) or 'line' in list(d.keys()):
            # Natural Oracle Format
            return d
    else:
        raise ValueError("json action dict is in an unknown format")

def parser1(d, env):
    action_space = env.action_space
    subs = set()
    for action in d:
        for sub_action in action['set_bus']['substations_id']:
            sub = sub_action[0]
            subs.add(sub)

    # init new dict with subs
    new_d = {'sub':{sub:[] for sub in subs}}

    # Pas bonne idée, parcourir dans la boucle
    grid = env.action_space.to_dict()

    for action in d:
        for sub_action in action['set_bus']['substations_id']:
            subid = sub_action[0]
            sub_topo = sub_action[1]

            # On cherche les ids des gens, loads et lines_ex/or modifiées par l'action sub_topo (qui donne le nouveau bus)
            # Generators
            gen_ids = [id_ for id_,subid_ in enumerate(grid['gen_to_subid']) if subid_ == subid] # id des générateurs concernés par cette substation
            new_action_on_gens = {"gens_id_bus":
                                      [[id_,sub_topo[grid['gen_to_sub_pos'][id_]]] for id_ in gen_ids] # Couples id du générateur, nouveau bus donné par sub_topo
                                  }
            # Loads
            load_ids = [id_ for id_, subid_ in enumerate(grid['load_to_subid']) if
                       subid_ == subid]
            new_action_on_loads = {"loads_id_bus":
                                      [[id_, sub_topo[grid['load_to_sub_pos'][id_]]] for id_ in load_ids]
                                  }
            # Lines origins and extremities gathered
            line_or_ids = [id_ for id_, subid_ in enumerate(grid['line_or_to_subid']) if
                        subid_ == subid]
            line_ex_ids = [id_ for id_, subid_ in enumerate(grid['line_ex_to_subid']) if
                           subid_ == subid]
            new_action_on_lines = {"lines_id_bus":
                                       [[id_, sub_topo[grid['line_or_to_sub_pos'][id_]]] for id_ in line_or_ids]+[[id_, sub_topo[grid['line_ex_to_sub_pos'][id_]]] for id_ in line_ex_ids]
                                   }
            new_action = {**new_action_on_loads,**new_action_on_gens,**new_action_on_lines}
            new_d['sub'][subid].append(new_action)
    # TODO: lines
    return new_d