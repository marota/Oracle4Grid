import json
import os
import numpy as np

from grid2op.Parameters import Parameters

from oracle4grid.core.utils.constants import EnvConstants
from oracle4grid.core.utils.prepare_environment import prepare_env
from oracle4grid.core.oracle import oracle


def load_and_run(env_dir, chronic, action_file, debug,agent_seed,env_seed, config, constants=EnvConstants()):
    atomic_actions, env, debug_directory = load(env_dir, chronic, action_file, debug, constants=constants)
    # Parse atomic_actions format
    atomic_actions = parse(atomic_actions,env)
    # Run all steps
    return oracle(atomic_actions, env, debug, config, debug_directory=debug_directory,agent_seed=agent_seed,env_seed=env_seed,
                  grid_path=env_dir, chronic_id=chronic, constants=constants)


def load(env_dir, chronic, action_file, debug, constants=EnvConstants()):
    param = Parameters()
    param.init_from_dict(constants.DICT_GAME_PARAMETERS_SIMULATION)
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
                print("Specific format is detected for actions: converting with parser 1")
                d = parser1(d,env)
                return d
    if type(d) is dict:
        if 'sub' in list(d.keys()) or 'line' in list(d.keys()):
            first_key = list(d.keys())[0]
            first_sub_or_line_id = list(d[first_key].keys())[0]
            if first_sub_or_line_id.isnumeric():
                if type(d[first_key][first_sub_or_line_id]) is dict:
                    specific_key = list(d[first_key][first_sub_or_line_id].keys())[0]
                    if specific_key == "set_configuration":
                        # Format 2 detected
                        print("Specific format is detected for actions: converting with parser 2")
                        d = parser2(d, env)
                        return d
                    elif type(d[first_key][first_sub_or_line_id]) is list:
                        # Natural Oracle Format
                        return d
                    else:
                        raise ValueError("json action dict is in an unknown format")
    else:
        raise ValueError("json action dict is in an unknown format")

def parser2(d, env):
    action_space = env.action_space
    new_dict = {line_or_sub:
                    {id_: [] for id_ in d[line_or_sub]}
                for line_or_sub in d.keys()}
    for line_or_sub in d:
        for id_ in d[line_or_sub]:
            action = d[line_or_sub][id_]['set_configuration']
            positions = np.argwhere(np.array(action) != 0)[:,0]
            for asset_pos_in_topo in positions:
                asset_action = action[asset_pos_in_topo]
                asset_type, asset_id = find_asset(action_space, asset_pos_in_topo) # TODO: développer find_asset
                new_dict[line_or_sub][id_] = create_or_update_asset_action(new_dict[line_or_sub][id_], asset_type, asset_id, asset_action)
    return new_dict

def find_asset(action_space, asset_pos_in_topo):
    # TODO:
    return

def create_or_update_asset_action(target_l, asset_type, asset_id, asset_action):
    found = False
    for i, action_d in enumerate(target_l):
        if list(action_d.keys())[0] == asset_type:
            found = True
            action_on_asset = action_d[asset_type].copy()
            action_on_asset.append([asset_id,asset_action])
            target_l[i][asset_type] = action_on_asset
    if not found:
        target_l.append({asset_type:[[asset_id,asset_action]]})
    return target_l


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