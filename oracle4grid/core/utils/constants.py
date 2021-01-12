from grid2op.Reward import L2RPNReward
from grid2op.Rules import AlwaysLegal


try:
    from lightsim2grid.LightSimBackend import LightSimBackend


    BACKEND = LightSimBackend
except ModuleNotFoundError:
    from grid2op.Backend import PandaPowerBackend


    BACKEND = PandaPowerBackend

# Grid2Op Env constants
REWARD_CLASS = L2RPNReward  # L2RPNSandBoxScore
GAME_RULE = AlwaysLegal
DICT_GAME_PARAMETERS_SIMULATION = {'NO_OVERFLOW_DISCONNECTION': True,
                                   'MAX_LINE_STATUS_CHANGED': 999,
                                   'MAX_SUB_CHANGED': 2999}
DICT_GAME_PARAMETERS_GRAPH = {'NO_OVERFLOW_DISCONNECTION': True,
                              'MAX_LINE_STATUS_CHANGED': 1,
                              'MAX_SUB_CHANGED': 1}
DICT_GAME_PARAMETERS_REPLAY = {'NO_OVERFLOW_DISCONNECTION': False,
                              'MAX_LINE_STATUS_CHANGED': 1,
                              'MAX_SUB_CHANGED': 1}

# Oracle constants
END_NODE_REWARD = 0.1

# Seed info
ENV_SEEDS = None
AGENT_SEEDS = None
