import unittest
import warnings
import json

from grid2op.Parameters import Parameters

from grid2op.Reward import L2RPNReward
from grid2op.Rules import AlwaysLegal
from oracle4grid.core.actions_utils import combinator
from oracle4grid.core.agent.OracleOverloadReward import OracleOverloadReward
from oracle4grid.core.reward_computation import run_many
from oracle4grid.core.utils.config_ini_utils import MAX_DEPTH, NB_PROCESS, MAX_ITER
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

CONFIG = {
    "max_depth": 3,
    "max_iter": 6,
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


class IntegrationTest(unittest.TestCase):
    def test_base_run(self):
        file = "./oracle4grid/ressources/actions/rte_case14_realistic/test_unitary_actions.json"
        chronic = "000"
        env_dir = "./data/rte_case14_realistic"
        action_path, grid2op_action_path, best_path_no_overload, grid2op_action_path_no_overload, kpis = load_and_run(env_dir, chronic, file, False, None, None, CONFIG, constants=EnvConstantsTest())
        self.assertNotEqual(action_path, None)
        return 1

    def test_best_path_equal_shortest(self):
        file = "./oracle4grid/ressources/actions/rte_case14_realistic/test_unitary_actions.json"
        chronic = "000"
        env_dir = "./data/rte_case14_realistic"
        action_path, grid2op_action_path, best_path_no_overload, grid2op_action_path_no_overload, indicators = load_and_run(env_dir, chronic, file, False, None, None, CONFIG, constants=EnvConstantsTest())
        best_path_actual = list(map(lambda x: x.__str__(), action_path))
        best_path_actual_no_overload = list(map(lambda x: x.__str__(), best_path_no_overload))
        best_path_expected = ['sub-5-2', 'sub-1-1_sub-5-2', 'sub-1-1_sub-5-2', 'sub-1-1_sub-5-2', 'sub-1-1_sub-5-2', 'sub-1-1_sub-5-2']
        best_path_expected_no_overload = ["sub-1-1", "sub-1-1", "sub-1-1", "sub-1-1", "sub-1-1", 'sub-1-1']
        self.assertListEqual(best_path_actual, best_path_expected)
        self.assertListEqual(best_path_actual_no_overload, best_path_expected_no_overload)

    def test_best_path_equal_longest(self):
        config_longest = CONFIG.copy()
        config_longest["best_path_type"] = "longest"

        file = "./oracle4grid/ressources/actions/rte_case14_realistic/test_unitary_actions.json"
        chronic = "000"
        env_dir = "./data/rte_case14_realistic"
        action_path, grid2op_action_path, best_path_no_overload, grid2op_action_path_no_overload, indicators = load_and_run(env_dir, chronic, file, False, None, None, config_longest)
        best_path_actual = list(map(lambda x: x.__str__(), action_path))
        best_path_actual_no_overload = list(map(lambda x: x.__str__(), best_path_no_overload))
        best_path_expected = ["line-4-3", "line-4-3", "line-4-3", "line-4-3", "line-4-3", "line-4-3"]
        best_path_expected_no_overload = best_path_expected
        self.assertListEqual(best_path_actual, best_path_expected)
        self.assertListEqual(best_path_actual_no_overload, best_path_expected_no_overload)

    def test_agent_rewards(self):
        # Compute best path with Oracle
        file = "./oracle4grid/ressources/actions/rte_case14_realistic/test_unitary_actions.json"
        chronic = 0
        env_dir = "./data/rte_case14_realistic"
        action_path, grid2op_action_path, best_path_no_overload, grid2op_action_path_no_overload, indicators = load_and_run(env_dir, chronic, file, False, None, None, CONFIG, constants=EnvConstantsTest())
        best_path_reward = float(indicators.loc[indicators[INDICATORS_NAMES_COL] == BEST_PATH_NAME, INDICATORS_REWARD_COL].values[0])

        # Replay path with OracleAgent as standard gym episode replay (OracleAgent not compatible with Grid2op Runner yet)
        constants = EnvConstantsTest()
        param = Parameters()
        param.init_from_dict(constants.DICT_GAME_PARAMETERS_GRAPH)
        env = prepare_env(env_dir, chronic, param, constants)
        env.set_id(chronic)
        obs = env.reset()
        agent = OracleAgent(action_path=grid2op_action_path, action_space=env.action_space,
                            observation_space=None, name=None)
        agent_reward = 0.
        done = False
        for t in range(CONFIG['max_iter']):
            if not done:
                action = agent.act(obs, reward=0., done=False)
                obs, reward, done, info = env.step(action)
                agent_reward += reward

        # Check if we get expected reward
        self.assertEqual(best_path_reward, agent_reward)

    expected_actions = ['sub-1-0',
                        'sub-1-1',
                        'sub-5-2',
                        'line-4-3',
                        'sub-1-0_sub-5-2',
                        'sub-1-1_sub-5-2',
                        'sub-1-0_line-4-3',
                        'sub-1-1_line-4-3',
                        'sub-5-2_line-4-3',
                        'sub-1-0_sub-5-2_line-4-3',
                        'sub-1-1_sub-5-2_line-4-3',
                        'donothing-0']

    def test_actions_combination(self):
        file = "./oracle4grid/ressources/actions/rte_case14_realistic/test_unitary_actions.json"
        chronic = "000"
        env_dir = "./data/rte_case14_realistic"
        atomic_actions, env, debug_directory = load(env_dir, chronic, file, False, constants=EnvConstantsTest())

        # 1 - Action generation step
        actions = combinator.generate(atomic_actions, int(CONFIG[MAX_DEPTH]), env, False)
        actions = map(lambda x: str(x), actions)
        self.assertListEqual(list(actions), self.expected_actions)

    def test_kpi(self):
        file = "./oracle4grid/ressources/actions/rte_case14_realistic/test_unitary_actions.json"
        chronic = "000"
        env_dir = "./data/rte_case14_realistic"
        action_path, grid2op_action_path, best_path_no_overload, grid2op_action_path_no_overload, indicators = load_and_run(env_dir, chronic, file, False, None, None, CONFIG, constants=EnvConstantsTest())
        expected = pd.read_csv('./oracle4grid/test_resourses/test_kpi.csv', sep=',', index_col=0)
        assert_frame_equal(indicators, expected)

    def test_reward_df(self):
        file = "./oracle4grid/ressources/actions/rte_case14_realistic/test_unitary_actions.json"
        chronic = "000"
        env_dir = "./data/rte_case14_realistic"
        atomic_actions, env, debug_directory = load(env_dir, chronic, file, False, constants=EnvConstantsTest())

        # 1 - Action generation step
        actions = combinator.generate(atomic_actions, int(CONFIG[MAX_DEPTH]), env, False)
        # 2 - Actions rewards simulation
        reward_df = run_many.run_all(actions, env, int(CONFIG[MAX_ITER]), int(CONFIG[NB_PROCESS]), debug=False)
        reward_df["action"] = reward_df["action"].astype(str)
        expected = pd.read_csv('./oracle4grid/test_resourses/test_reward.csv', sep=',', index_col=0)
        assert_frame_equal(reward_df, expected)

    def test_action_divergence(self):
        # Compute best path with Oracle
        file = "./oracle4grid/ressources/actions/rte_case14_realistic/test_unitary_actions.json"
        chronic = 0
        env_dir = "./data/rte_case14_realistic"

        # Check that replay returns warning because of non convergence of one action
        with warnings.catch_warnings(record=True) as w:
            action_path, grid2op_action_path, best_path_no_overload, grid2op_action_path_no_overload, indicators = load_and_run(env_dir, chronic, file, False, None, None, CONFIG, constants=EnvConstantsTest())
            boolvec_msg = ["During replay - oracle agent has game over before max iter" in str(w_.message) for w_ in w]
            self.assertTrue(np.any(boolvec_msg))

    def test_action_convergence(self):
        config_longest = CONFIG.copy()
        config_longest["best_path_type"] = "longest"

        # Compute best path with Oracle
        file = "./oracle4grid/ressources/actions/rte_case14_realistic/test_unitary_actions.json"
        chronic = 0
        env_dir = "./data/rte_case14_realistic"

        # Check that replay returns warning because of non convergence of one action
        with warnings.catch_warnings(record=True) as w:
            action_path, grid2op_action_path, best_path_no_overload, grid2op_action_path_no_overload, indicators = load_and_run(env_dir, chronic, file, False, None, None,
                                                                        config_longest)
            boolvec_msg = ["During replay - oracle agent has game over before max iter" in str(w_.message) for w_ in w]
            if len(boolvec_msg) == 0:  # Test is OK if no warning
                boolvec_msg = [False]
            self.assertFalse(np.all(boolvec_msg))

    def test_parsing1(self):
        file_json = "./oracle4grid/test_resourses/test_actions_format1.json"
        chronic = 0
        env_dir = "./data/wcci_test"

        # Load env
        param = Parameters()
        param.init_from_dict(EnvConstantsTest().DICT_GAME_PARAMETERS_SIMULATION)
        env = prepare_env(env_dir, chronic, param)

        # Read and convert action
        with open(file_json) as json_file:
            actions_original = json.load(json_file)
        parser = OracleParser(actions_original, env.action_space)
        action_oracle_format = parser.parse()

        # Read expected format
        with open("./oracle4grid/test_resourses/expected_actions_format1.json") as json_file_expected:
            expected_format = json.load(json_file_expected)

        # Assert Equal
        self.assertEqual(action_oracle_format, expected_format)

    def test_parsing2(self):
        file_json = "./oracle4grid/test_resourses/test_actions_format2.json"
        chronic = 0
        env_dir = "./data/wcci_test"

        # Load env
        param = Parameters()
        param.init_from_dict(EnvConstantsTest().DICT_GAME_PARAMETERS_SIMULATION)
        env = prepare_env(env_dir, chronic, param)

        # Read and convert action
        with open(file_json) as json_file:
            actions_original = json.load(json_file)
        parser = OracleParser(actions_original, env.action_space)
        action_oracle_format = parser.parse()

        # Read expected format
        with open("./oracle4grid/test_resourses/expected_actions_format2.json") as json_file_expected:
            expected_format = json.load(json_file_expected)

        # Assert Equal
        self.assertEqual(action_oracle_format, expected_format)

    def test_actions_impact(self):
        file_json = "./oracle4grid/test_resourses/test_actions_format2.json"
        chronic = 0
        env_dir = "./data/wcci_test"

        atomic_actions_original, env, debug_directory = load(env_dir, chronic, file_json, debug=False, constants=EnvConstantsTest())
        parser = OracleParser(atomic_actions_original, env.action_space)
        atomic_actions = parser.parse()

        actions = combinator.generate(atomic_actions, 1, env, False)
        impacted_subs = []
        expected_impacted_subs = [1, 23, 16, 22, 26]
        for action in actions[:-1]: # Do nothing action is the last one
            impact_on_subs = action.grid2op_action.get_topological_impact()[1]
            impacted_sub = int(np.where(impact_on_subs)[0])
            impacted_subs.append(impacted_sub)
        self.assertEqual(impacted_subs, expected_impacted_subs)



if __name__ == '__main__':
    unittest.main()
