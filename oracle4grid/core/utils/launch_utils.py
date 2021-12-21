import json
import os
import numpy as np

from grid2op.Parameters import Parameters
from oracle4grid.core.utils.config_ini_utils import MAX_ITER

from oracle4grid.core.utils.constants import EnvConstants
from oracle4grid.core.utils.prepare_environment import prepare_env
from oracle4grid.core.oracle import oracle

ASSET_MAPPING = {"line (origin)":"lines_id_bus",
                 "line (extremity)":"lines_id_bus",
                 "generator":"gens_id_bus",
                 "load":"loads_id_bus"}

def load_and_run(env_dir, chronic, action_file, debug,agent_seed,env_seed, config, constants=EnvConstants()):
    atomic_actions, env, debug_directory, chronic_id = load(env_dir, chronic, action_file, debug, constants=constants, config = config)
    # Parse atomic_actions format
    # atomic_actions = parse(atomic_actions,env)
    parser = OracleParser(atomic_actions, env.action_space)
    atomic_actions = parser.parse()

    # Run all steps
    return oracle(atomic_actions, env, debug, config, debug_directory=debug_directory,agent_seed=agent_seed,env_seed=env_seed,
                  grid_path=env_dir, chronic_scenario=chronic, constants=constants)


def load(env_dir, chronic, action_file, debug, constants=EnvConstants(), config = None, opponent_allowed=True):
    param = Parameters()
    param.init_from_dict(constants.DICT_GAME_PARAMETERS_SIMULATION)
    env, chronic_id = prepare_env(env_dir, chronic, param, opponent_allowed=opponent_allowed)

    # Load unitary actions
    with open(action_file) as f:
        atomic_actions = json.load(f)

    # Init debug mode if necessary
    if debug:
        try:
            output_path = config["output_path"]
        except:
            output_path = "oracle4grid/output" # os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..",'output')
        debug_directory = init_debug_directory(env_dir, action_file, chronic, output_path)
    else:
        debug_directory = None
    return atomic_actions, env, debug_directory, chronic_id


def init_debug_directory(env_dir, action_file, chronic, output_path = None):

    action_file_os = os.path.split(action_file)[len(os.path.split(action_file)) - 1].replace(".json", "")
    grid_file_os = os.path.split(env_dir)[len(os.path.split(env_dir)) - 1]
    scenario = "scenario_" + str(chronic)
    debug_directory = os.path.join(output_path, grid_file_os, scenario, action_file_os)
    os.makedirs(debug_directory, exist_ok=True)
    replay_debug_directory = os.path.join(debug_directory, "replay_logs")
    os.makedirs(replay_debug_directory, exist_ok=True)
    return debug_directory

class OracleParser():
    def __init__(self, d, action_space):
        self.d = d
        self.action_space = action_space
        self.parse = self.choose_parser_function()

    def choose_parser_function(self):
        if type(self.d) is list:
            if 'set_bus' in list(self.d[0].keys()):
                if 'substations_id' in list(self.d[0]['set_bus'].keys()):
                    # Format 1 detected
                    print("Specific format is detected for actions: converting with parser 1")
                    return self.parser1
        if type(self.d) is dict:
            if 'sub' in list(self.d.keys()) or 'line' in list(self.d.keys()):
                first_key = list(self.d.keys())[0]
                first_sub_or_line_id = list(self.d[first_key].keys())[0]
                if first_sub_or_line_id.isnumeric():
                    if type(self.d[first_key][first_sub_or_line_id]) is list:
                        first_action = self.d[first_key][first_sub_or_line_id][0]
                        specific_key = list(first_action.keys())[0]
                        if specific_key == "set_configuration":
                            # Format 2 detected
                            print("Specific format is detected for actions: converting with parser 2")
                            return self.parser2
                        elif specific_key in list(ASSET_MAPPING.values()) or specific_key == "set_line":
                            # Natural Oracle Format
                            print("Natural Oracle format is detected for actions")
                            return self.parser0
                        else:
                            raise ValueError("json action dict is in an unknown format - action key "+str(specific_key)+" not handled")
                    else:
                        raise ValueError("json action dict is in an unknown format")
                else:
                    raise ValueError("json action dict is in an unknown format")
        else:
            raise ValueError("json action dict is in an unknown format")

    def parser0(self):
        return self.d

    def parser1(self):
        subs = set()
        for action in self.d:
            for sub_action in action['set_bus']['substations_id']:
                sub = sub_action[0]
                subs.add(sub)

        # init new dict with subs
        new_d = {'sub': {sub: [] for sub in subs}}

        # Pas bonne idée, parcourir dans la boucle
        grid = self.action_space.cls_to_dict()

        for action in self.d:
            for sub_action in action['set_bus']['substations_id']:
                subid = sub_action[0]
                sub_topo = sub_action[1]

                # On cherche les ids des gens, loads et lines_ex/or modifiées par l'action sub_topo (qui donne le nouveau bus)
                # Generators
                gen_ids = [id_ for id_, subid_ in enumerate(grid['gen_to_subid']) if
                           subid_ == subid]  # id des générateurs concernés par cette substation
                new_action_on_gens = {"gens_id_bus":
                                          [[id_, sub_topo[grid['gen_to_sub_pos'][id_]]] for id_ in gen_ids]
                                      # Couples id du générateur, nouveau bus donné par sub_topo
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
                                           [[id_, sub_topo[grid['line_or_to_sub_pos'][id_]]] for id_ in line_or_ids] + [
                                               [id_, sub_topo[grid['line_ex_to_sub_pos'][id_]]] for id_ in line_ex_ids]
                                       }
                new_action = {**new_action_on_loads, **new_action_on_gens, **new_action_on_lines}
                new_d['sub'][subid].append(new_action)
        # TODO: lines
        return new_d

    def parser2(self):
        new_dict = {line_or_sub:
                        {id_: [] for id_ in self.d[line_or_sub]}
                    for line_or_sub in self.d.keys()}
        for line_or_sub in self.d:
            for id_ in self.d[line_or_sub]:
                for original_action in self.d[line_or_sub][id_]:
                    action = np.array(original_action['set_configuration'])
                    asset_types, asset_ids, asset_actions = find_and_check_action_on_assets(action, self.action_space,
                                                                                            line_or_sub, int(id_))
                    unitary_action_dict = get_unitary_action_dict(asset_types, asset_ids, asset_actions, line_or_sub)
                    target_l = new_dict[line_or_sub][id_].copy()
                    target_l.append(unitary_action_dict)
                    new_dict[line_or_sub][id_] = target_l
        return new_dict

def find_and_check_action_on_assets(action, action_space, line_or_sub, id_):
    impact = action_space.from_vect(action).impact_on_objects()

    # Initialize list of results
    asset_types = []
    asset_ids = []
    asset_actions = []

    # In case the action is on sub, check it is the case and on the right sub
    # Then, extract infos on assets impacted
    if line_or_sub == 'sub':
        bus_impact = impact['topology']['assigned_bus']
        if len(bus_impact) == 0:
            raise ValueError("Declared sub action on sub number"+str(id_)+" doesnt impact substation bus")
        else:
            for sub_action in bus_impact:
                if sub_action['substation'] != id_:
                    raise ValueError("Declared sub action on sub number"+str(id_)+" impacts an other substation (sub number "+str(sub_action['substation'])+")")
                else:
                    asset_actions.append(int(sub_action['bus']))
                    asset_ids.append(int(sub_action['object_id']))
                    asset_types.append(ASSET_MAPPING[sub_action['object_type']])

    # In case it is line disconnection, just check it impacts the right line
    elif line_or_sub == "line":
        line_impact = impact['force_line']['disconnections']['powerlines']
        if len(line_impact) == 0:
            raise ValueError("Declared line action on line number"+str(id_)+" doesnt disconnect any line")
        else:
            for line_id_disc in line_impact:
                if line_id_disc != id_:
                    raise ValueError("Declared line disconnection on line number"+str(id_)+" impacts an other line (sub number "+str(line_id_disc)+")")

    return asset_types, asset_ids, asset_actions


def get_unitary_action_dict(asset_types, asset_ids, asset_actions, line_or_sub):
    if line_or_sub == "sub":
        d = dict()
        for asset_type, asset_id, asset_action in zip(asset_types, asset_ids, asset_actions):
            if asset_type in list(d.keys()):
                # update new sub action on this asset type
                action_on_asset = d[asset_type].copy()
                action_on_asset.append([asset_id, asset_action])
                d[asset_type] = action_on_asset
            else:
                # First action on this asset type
                d[asset_type] = [[asset_id, asset_action]]
    elif line_or_sub == "line":
        d = {"set_line":-1}
    return d
