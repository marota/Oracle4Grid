import time
import unittest
import warnings
import json

from grid2op.Parameters import Parameters

from grid2op.Reward import L2RPNReward
from grid2op.Rules import AlwaysLegal
from oracle4grid.core.actions_utils import combinator
from oracle4grid.core.agent.OracleOverloadReward import OracleOverloadReward
from oracle4grid.core.graph import graph_generator
from oracle4grid.core.reward_computation import run_many
from oracle4grid.core.utils.config_ini_utils import MAX_DEPTH, NB_PROCESS, MAX_ITER, REWARD_SIGNIFICANT_DIGIT
from oracle4grid.core.utils.constants import EnvConstants
from oracle4grid.core.utils.launch_utils import load_and_run, load
from oracle4grid.core.agent.OracleAgent import OracleAgent
from oracle4grid.core.utils.prepare_environment import prepare_env, get_initial_configuration
from oracle4grid.core.utils.launch_utils import OracleParser


from pandas.testing import assert_frame_equal
import pandas as pd
import numpy as np
import os


BEST_PATH_NAME = "Best possible path with game rules"
INDICATORS_NAMES_COL = "Indicator name"
INDICATORS_REWARD_COL = "Reward value"

config = {
    "max_depth": 5,
    "max_iter": 100,
    "nb_process": 1,
    "best_path_type": "shortest",
    "n_best_topos": 2,
    "reward_significant_digit": 2
}


class EnvConstantsTest(EnvConstants):
    def __init__(self):
        super().__init__()
        self.reward_class = L2RPNReward
        self.other_rewards = {
            "overload_reward": OracleOverloadReward
        }
        self.game_rule = AlwaysLegal
        self.DICT_GAME_PARAMETERS_SIMULATION = {'NO_OVERFLOW_DISCONNECTION': True,
                                                'MAX_LINE_STATUS_CHANGED': 999,
                                                'MAX_SUB_CHANGED': 2999}
        self.DICT_GAME_PARAMETERS_GRAPH = {'NO_OVERFLOW_DISCONNECTION': True,
                                           'MAX_LINE_STATUS_CHANGED': 1,
                                           'MAX_SUB_CHANGED': 1}
        self.DICT_GAME_PARAMETERS_REPLAY = {'NO_OVERFLOW_DISCONNECTION': False,
                                            'MAX_LINE_STATUS_CHANGED': 1,
                                            'MAX_SUB_CHANGED': 1}


class PerformanceTest(unittest.TestCase):
    def test_graph_duration(self):
        file = "./oracle4grid/ressources/actions/rte_case14_realistic/test_unitary_actions.json"
        chronic = "000"
        env_dir = "./data/rte_case14_realistic"
        atomic_actions, env, debug_directory, chronic_id = load(env_dir, chronic, file, False, constants=EnvConstantsTest())
        parser = OracleParser(atomic_actions, env.action_space)
        atomic_actions = parser.parse()
        # 0 - Preparation : Get initial topo and line status
        init_topo_vect, init_line_status = get_initial_configuration(env)

        # 1 - Action generation step
        start_time = time.time()
        actions = combinator.generate(atomic_actions, int(config[MAX_DEPTH]), env, False, nb_process=int(config[NB_PROCESS]))
        elapsed_time = time.time() - start_time
        print("elapsed_time for action generation is:"+str(elapsed_time))

        # 2 - Actions rewards simulation
        start_time = time.time()
        reward_df = run_many.run_all(actions, env, int(config[MAX_ITER]), int(config[NB_PROCESS]), debug=False,
                                     agent_seed=None,env_seed=None)
        elapsed_time = time.time() - start_time
        print("elapsed_time for simulation is:"+str(elapsed_time))

        # 3 - Graph generation
        start_time = time.time()
        graph = graph_generator.generate(reward_df, int(config[MAX_ITER])
                                         , debug=False,reward_significant_digit=config[REWARD_SIGNIFICANT_DIGIT], constants=EnvConstantsTest())

        elapsed_time = time.time() - start_time
        print("elapsed_time for graph creation is:"+str(elapsed_time))
        assert elapsed_time < 3
        return 1


if __name__ == '__main__':
    unittest.main()
