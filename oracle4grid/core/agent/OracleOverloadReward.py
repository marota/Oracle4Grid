import numpy as np
from grid2op.Reward.BaseReward import BaseReward
from grid2op.dtypes import dt_float


class OracleOverloadReward(BaseReward):
    def __init__(self):
        BaseReward.__init__(self)

    def initialize(self, env):
        self.reward_min = -1
        self.reward_max = 1

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error: # or is_done
            return -1
        elif has_overflow(env):
            return 0
        else:
            return 1


def has_overflow(env):
    return rho(env)


def rho(env):
    ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
    thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
    thermal_limits += 1e-1  # for numerical stability
    relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)
    return (relative_flow > 1).any()
