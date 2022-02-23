import os
import numpy as np
from grid2op.Action.DontAct import DontAct
from grid2op.Action import PlayableAction

import grid2op
from grid2op.Chronics import GridStateFromFile
from oracle4grid.core.utils.constants import EnvConstants, BACKEND
from grid2op import make
from grid2op.Chronics import FromNPY


def prepare_env(env_path, chronic_scenario, param, constants=EnvConstants(), opponent_allowed=True):
    backend = BACKEND()
    env = None
    if opponent_allowed:
        env = grid2op.make(env_path,
                           reward_class=constants.reward_class,
                           backend=backend,
                           data_feeding_kwargs={"gridvalueClass": GridStateFromFile},
                           param=param,
                           gamerules_class=constants.game_rule,
                           test=True,# Why test = True is used here ?
                           other_rewards=constants.other_rewards,
                           # We need the actions of the agent to be the highest base class
                           action_class=PlayableAction
                           )
    else:
        env = grid2op.make(env_path,
                           reward_class=constants.reward_class,
                           backend=backend,
                           data_feeding_kwargs={"gridvalueClass": GridStateFromFile},
                           param=param,
                           gamerules_class=constants.game_rule,
                           test=True,
                           other_rewards=constants.other_rewards,
                           # We need the actions of the agent to be the highest base class
                           action_class=PlayableAction,
                           opponent_init_budget=0,
                           opponent_action_class=DontAct
                           )

    # If an int is provided, chronic_scenario is string by default, so it has to be converted
    try:
        chronic_scenario = int(chronic_scenario)
        print(
            "INFO - An integer has been provided as chronic scenario - looking for the chronic folder in this position")
    except:
        if chronic_scenario is None:
            print("INFO - No value has been provided for chronic scenario - the first chronic folder will be chosen")
        else:
            print(
                "INFO - A string value has been provided as chronic scenario - searching for a chronic folder with name " + str(
                    chronic_scenario))

    # Go to desired chronic scenario (if None, first scenario will be taken)
    if chronic_scenario is not None:
        if type(chronic_scenario) is str:
            found_id = search_chronic_num_from_name(chronic_scenario, env)
        elif type(chronic_scenario) is int:
            found_id = chronic_scenario
            scenario_name = search_chronic_name_from_num(found_id, env)
            print("INFO - the name of the loaded Grid2op scenario is : " + str(scenario_name))
        if found_id is not None:
            env.set_id(found_id)
            env.reset()
        else:  # if name not found
            raise ValueError("Chronic scenario name: " + str(chronic_scenario) + " not found in folder")
    return env,found_id

def create_env_late_start_multivers(env_ref, begin_time,end_time, constants=EnvConstants(), opponent_allowed=True):
    load_p = 1.0 * env_ref.chronics_handler.real_data.data.load_p
    load_q = 1.0 * env_ref.chronics_handler.real_data.data.load_q
    prod_p = 1.0 * env_ref.chronics_handler.real_data.data.prod_p
    prod_v = 1.0 * env_ref.chronics_handler.real_data.data.prod_v
    maintenance = env_ref.chronics_handler.real_data.data.maintenance

    backend = BACKEND()

    # now create an environment with these chronics:
    #no opponent for multiverse, we play attacks by "hand"
    env = make(env_ref.get_path_env(),#env_ref.name +"_"+ str(begin_time),
               reward_class=constants.reward_class,
               backend=backend,
               param=env_ref.parameters,
               gamerules_class=constants.game_rule,
               #test=True,
               other_rewards=constants.other_rewards,
               # We need the actions of the agent to be the highest base class
               action_class=PlayableAction,
               opponent_init_budget=0,
               opponent_action_class=DontAct,
               chronics_class=FromNPY,
               data_feeding_kwargs={"i_start": begin_time,
                                    # start at the "step" 5 NB first step is first observation, available with `obs = env.reset()`
                                    "i_end": end_time,
                                    # end index: data after that will not be considered (excluded as per python convention)
                                    "load_p": load_p,
                                    "load_q": load_q,
                                    "prod_p": prod_p,
                                    "prod_v": prod_v,
                                    # other parameters includes
                                    "maintenance": maintenance,
                                    # load_p_forecast
                                    # load_q_forecast
                                    # prod_p_forecast
                                    # prod_v_forecast
                                    "gridvalueClass": GridStateFromFile
                                    })

    return env


def search_chronic_name_from_num(num, env):
    for id, sp in enumerate(env.chronics_handler.real_data.subpaths):
        chronic_scenario = os.path.basename(sp)
        if id == num:
            break
    return chronic_scenario


def search_chronic_num_from_name(scenario_name, env):
    found_id = None
    # Search scenario with provided name
    for id, sp in enumerate(env.chronics_handler.real_data.subpaths):
        sp_end = os.path.basename(sp)
        if sp_end == scenario_name:
            found_id = id
    return found_id


def get_initial_configuration(env):
    init_topo_vect = env.get_obs().topo_vect
    init_line_status = env.get_obs().line_status * 1
    init_line_status = np.where(init_line_status == 0, -1, init_line_status)
    return init_topo_vect, init_line_status
