from grid2op.Reward import L2RPNReward
from grid2op.Rules import AlwaysLegal
try:
    from lightsim2grid import LightSimBackend
    BACKEND = LightSimBackend
except:
    from grid2op.Backend import PandaPowerBackend
    BACKEND = PandaPowerBackend

# Grid2Op Env constants
REWARD_CLASS = L2RPNReward # L2RPNSandBoxScore
GAME_RULE = AlwaysLegal
DICT_GAME_PARAMETERS = {'NO_OVERFLOW_DISCONNECTION': True,
                        'MAX_LINE_STATUS_CHANGED': 999,
                        'MAX_SUB_CHANGED': 2999}