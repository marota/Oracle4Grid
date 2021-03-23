import grid2op
from grid2op.Parameters import Parameters
from grid2op.Chronics import GridStateFromFile
try:
    from lightsim2grid.LightSimBackend import LightSimBackend
    BACKEND = LightSimBackend
except ModuleNotFoundError:
    from grid2op.Backend import PandaPowerBackend
    BACKEND = PandaPowerBackend

env_path='/Users/antoinemarot/Grid2Op_EnvironmentDesign/competition_codalab/L2RPN_neurips2020_track1/input_data_test'

param_simu = Parameters()
param_simu.init_from_dict({'NO_OVERFLOW_DISCONNECTION': True,
                                           'MAX_LINE_STATUS_CHANGED': 999,
                                           'MAX_SUB_CHANGED': 2999})

param_replay = Parameters()
param_replay.init_from_dict({'NO_OVERFLOW_DISCONNECTION': False,
                                       'MAX_LINE_STATUS_CHANGED': 1,
                                       'MAX_SUB_CHANGED': 1})

env_simu=backend = grid2op.make(env_path,
                       backend=BACKEND,
                       data_feeding_kwargs={"gridvalueClass": GridStateFromFile},
                       param=param_simu,
                       test=True,
                       )
env_replay=backend = grid2op.make(env_path,
                       backend=BACKEND,
                       data_feeding_kwargs={"gridvalueClass": GridStateFromFile},
                       param=param_replay,
                       test=True,
                       )


action_sub_26={'set_bus': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 2,
        2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0]}

obs_simu=env_simu.reset()
obs, reward, done, info_simu=obs_simu.simulate(env_simu.action_space(action_sub_26),time_step=0)

obs_replay = env_replay.reset()
obs, reward, done, info_replay=obs_replay.simulate(env_simu.action_space(action_sub_26),time_step=0)

print("debug")