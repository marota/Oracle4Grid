#outputs a dataframe with rewards for each run (run = N actions) and each timestep
#takes a dict of all combinations of actions
import grid2op
from grid2op.Chronics import GridStateFromFile
from grid2op.Parameters import Parameters
from grid2op.Rules import AlwaysLegal
from lightsim2grid.LightSimBackend import LightSimBackend

from core.utils.constants import *


def run_all(actions, ini, env_path) :
    param = prepareParams(env_path, ini)
    env = prepareEnv(param)
    df = parallel(env, actions, ini)
    return df

def prepareParams(ini) :
    param = Parameters()
    #TODO: from ini file
    param.init_from_dict({'NO_OVERFLOW_DISCONNECTION': True})
    param.init_from_dict({'MAX_LINE_STATUS_CHANGED': 999})
    param.init_from_dict({'MAX_SUB_CHANGED': 2999})
    return param

def prepareEnv(env_path, param) :
    backend = LightSimBackend()
    return grid2op.make(env_path,
                       reward_class=REWARD_CLASS,
                       backend=backend,
                       data_feeding_kwargs={"gridvalueClass": GridStateFromFile},
                       param=param,
                       gamerules_class=GAME_RULE,
                       test=True,
                       )

def parallel(env, actions, ini) :
    #run in parallel all actions and return df of rewards indexed by timestep, atomic actions and subs
    #["time", "depth1", "depth2", ... , "depthn", "grid2opAction", "sub", "reward"]