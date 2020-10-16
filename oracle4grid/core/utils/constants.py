from grid2op.Backend import PandaPowerBackend
from grid2op.Reward import L2RPNReward
from grid2op.Rules import AlwaysLegal

# Grid2Op Env constants
REWARD_CLASS = L2RPNReward
GAME_RULE = AlwaysLegal
BACKEND = LightSimBackend
# BACKEND = PandaPowerBackend