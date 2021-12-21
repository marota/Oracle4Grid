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
    "reward_significant_digit": 2,
    "replay_reward_rel_tolerance":1e7
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

        # Replay path with OracleAgent as standard gym episode replay
        constants = EnvConstantsTest()
        param = Parameters()
        param.init_from_dict(constants.DICT_GAME_PARAMETERS_GRAPH)
        env, chronic_id = prepare_env(env_dir, chronic, param, constants)
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

    def test_agent_reco(self):
        # Parameters
        timestep_disconnect = 2
        total_timesteps = 4
        line_to_disconnect = 6
        chronic = 0
        env_dir = "./data/rte_case14_realistic"

        # Load env
        constants = EnvConstantsTest()
        param = Parameters()
        param.init_from_dict(constants.DICT_GAME_PARAMETERS_GRAPH)
        env, chronic_id = prepare_env(env_dir, chronic, param, constants)
        env.set_id(chronic)
        obs = env.reset()

        # Compute a fake Oracle action path and Oracle Agent
        grid2op_action = env.action_space({"set_line_status":[(9,-1)]})
        grid2op_action_path = [grid2op_action for t in range(total_timesteps)]
        agent = OracleAgent(action_path=grid2op_action_path, action_space=env.action_space,
                            observation_space=None, name=None)

        # Play OracleAgent and disconnect a line - test if it is well reconnected by agent
        action_disconnect = env.action_space({"set_line_status":[(line_to_disconnect,-1)]})
        done = False
        for t in range(total_timesteps):
            if t == timestep_disconnect:
                obs, reward, done, info = env.step(action_disconnect)
            else:
                if not done:
                    action = agent.act(obs, reward=0., done=False)
                    obs, reward, done, info = env.step(action)

        # Check if we get expected reward
        self.assertEqual(obs.line_status[line_to_disconnect], True)

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
        atomic_actions, env, debug_directory, chronic_id = load(env_dir, chronic, file, False, constants=EnvConstantsTest())

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
        atomic_actions, env, debug_directory, chronic_id = load(env_dir, chronic, file, False, constants=EnvConstantsTest())

        cols_to_check = ['action', 'timestep', 'reward', 'overload_reward', 'attack_id']

        # 1 - Action generation step
        actions = combinator.generate(atomic_actions, int(CONFIG[MAX_DEPTH]), env, False)
        # 2 - Actions rewards simulation
        reward_df = run_many.run_all(actions, env, int(CONFIG[MAX_ITER]), int(CONFIG[NB_PROCESS]), debug=False)
        reward_df["action"] = reward_df["action"].astype(str)
        expected = pd.read_csv('./oracle4grid/test_resourses/test_reward.csv', sep=',', index_col=0)
        expected['attack_id'] = expected['attack_id'].astype(float)
        reward_df['attack_id'] = reward_df['attack_id'].astype(float)
        assert_frame_equal(reward_df[cols_to_check], expected[cols_to_check])

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
        env, chronic_id = prepare_env(env_dir, chronic, param)

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
        chronic = 0
        env_dir = "./data/wcci_test"

        # Load env
        param = Parameters()
        param.init_from_dict(EnvConstantsTest().DICT_GAME_PARAMETERS_SIMULATION)
        env, chronic_id = prepare_env(env_dir, chronic, param)
        # Read expected format
        with open("./oracle4grid/test_resourses/expected_actions_format2.json") as json_file_expected:
            expected_format = json.load(json_file_expected)

        # Create the json format based on to_vect
        actions = combinator.generate(expected_format, 1, env, False, nb_process=1)
        vect_format = {"sub": {}}
        for action in actions:
            if len(action.subs.keys()) > 0:
                vect_format["sub"][str(list(action.subs.keys())[0])] = [{'set_configuration':action.grid2op_action.to_vect()}]

        #parse this format 2
        parser = OracleParser(vect_format, env.action_space)
        action_oracle_format = parser.parse()

        # Assert Equal
        self.assertEqual(action_oracle_format, expected_format)

    def test_actions_impact(self):
        file_json = "./oracle4grid/test_resourses/expected_actions_format2.json"
        chronic = 0
        env_dir = "./data/wcci_test"

        atomic_actions_original, env, debug_directory, chronic_id = load(env_dir, chronic, file_json, debug=False, constants=EnvConstantsTest())
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

    def test_no_path_without_overload(self):
        config_longest = CONFIG.copy()
        config_longest["best_path_type"] = "shortest"

        file = "./oracle4grid/test_resourses/test_unitary_actions_overload.json"
        chronic = "000"
        env_dir = "./oracle4grid/test_resourses/grids/rte_case14_realistic_overload"
        action_path, grid2op_action_path, best_path_no_overload, grid2op_action_path_no_overload, indicators = load_and_run(
            env_dir, chronic, file, False, None, None, config_longest)
        best_path_actual = list(map(lambda x: x.__str__(), action_path))
        best_path_actual_no_overload = list(map(lambda x: x.__str__(), best_path_no_overload))

        best_path_expected = ['donothing-0', 'donothing-0', 'donothing-0', 'donothing-0', 'donothing-0', 'donothing-0']
        best_path_expected_no_overload = []
        self.assertListEqual(best_path_actual, best_path_expected)
        self.assertListEqual(best_path_actual_no_overload, best_path_expected_no_overload)

    def test_cancelling_action(self):
        # Parameters
        chronic = 0
        env_dir = "./data/rte_case14_realistic"
        file_json = "./oracle4grid/test_resourses/test_unitary_actions_cancelling.json"

        # Compute a fake Oracle action path and Oracle Agent
        atomic_actions_original, env, debug_directory, chronic_id = load(env_dir, chronic, file_json, debug=False,
                                                             constants=EnvConstantsTest())
        obs = env.reset()
        init_topo_vect, init_line_status = get_initial_configuration(env)
        parser = OracleParser(atomic_actions_original, env.action_space)
        atomic_actions = parser.parse()
        actions = combinator.generate(atomic_actions, 2, env, False)

        action_path = [actions[0],actions[2], actions[1]]
        grid2op_action_path = [action.grid2op_action for action in action_path]

        agent = OracleAgent(action_path=grid2op_action_path, action_space=env.action_space,
                            oracle_action_path=action_path,
                            observation_space=None, name=None,
                            init_line_status=init_line_status, init_topo_vect=init_topo_vect)

        # Play OracleAgent and disconnect a line - test if it is well reconnected by agent
        done = False
        for t in range(3):
            if not done:
                action = agent.act(obs, reward=0., done=False)
                obs, reward, done, info = env.step(action)

        # Check if there is the right cancelling action
        impact = action.impact_on_objects()['topology']["assigned_bus"]
        expected_impact = [{'bus': 1,
                             'object_type': 'line (extremity)',
                             'object_id': 0,
                             'substation': 1},
                            {'bus': 1, 'object_type': 'line (origin)', 'object_id': 2, 'substation': 1},
                            {'bus': 1, 'object_type': 'load', 'object_id': 0, 'substation': 1},
                            {'bus': 2,
                            'object_type': 'line (origin)',
                            'object_id': 7,
                            'substation': 5},
                           {'bus': 2,
                            'object_type': 'line (extremity)',
                            'object_id': 17,
                            'substation': 5}
                            ]

        self.assertListEqual(impact, expected_impact)

    def test_cancelling_action_sub1(self):
        # Parameters
        chronic = 0
        env_dir = "./data/rte_case14_realistic"
        file_json = "./oracle4grid/test_resourses/test_unitary_actions_cancelling_sub1.json"


        # Compute a fake Oracle action path and Oracle Agent
        atomic_actions_original, env, debug_directory, chronic_id = load(env_dir, chronic, file_json, debug=False,
                                                             constants=EnvConstantsTest())
        init_topo_vect, init_line_status = get_initial_configuration(env)
        obs = env.reset()
        parser = OracleParser(atomic_actions_original, env.action_space)
        atomic_actions = parser.parse()
        actions = combinator.generate(atomic_actions, 2, env, False)

        action_path = [actions[0],actions[1]] # sub-1-0 and then sub-1-1
        grid2op_action_path = [action.grid2op_action for action in action_path]

        agent = OracleAgent(action_path=grid2op_action_path, action_space=env.action_space,
                            oracle_action_path=action_path,
                            observation_space=None, name=None,
                            init_line_status=init_line_status, init_topo_vect=init_topo_vect)

        # Play OracleAgent and disconnect a line - test if it is well reconnected by agent
        done = False
        for t in range(2):
            if not done:
                action = agent.act(obs, reward=0., done=False)
                obs, reward, done, info = env.step(action)

        # Check if there is the right cancelling action
        impact = action.impact_on_objects()['topology']["assigned_bus"]
        expected_impact = [{'bus': 1,
                          'object_type': 'line (extremity)',
                          'object_id': 0,
                          'substation': 1},
                         {'bus': 2, 'object_type': 'line (origin)', 'object_id': 2, 'substation': 1},
                         {'bus': 2, 'object_type': 'load', 'object_id': 0, 'substation': 1}] # We test that load 0 has been set to bus 2 and that there is no ambiguosity

        self.assertListEqual(impact, expected_impact)

    def test_run_oracleagent_randompath(self):
        # Parameters
        chronic = 0
        env_dir = "./data/rte_case14_realistic"
        file_json = "./oracle4grid/test_resourses/test_unitary_actions_final_topo.json"
        max_depth = 4
        max_timestep = 100

        # Compute a fake Oracle action path and Oracle Agent
        atomic_actions_original, env, debug_directory, chronic_id = load(env_dir, chronic, file_json, debug=False,
                                                             constants=EnvConstantsTest())
        init_topo_vect, init_line_status = get_initial_configuration(env)
        obs = env.reset()
        parser = OracleParser(atomic_actions_original, env.action_space)
        atomic_actions = parser.parse()
        actions = combinator.generate(atomic_actions, max_depth, env, False)

        # Compute an abritrary path with traps
        np.random.seed(1000)
        action_path = np.random.choice(actions, max_timestep).tolist()
        grid2op_action_path = [action.grid2op_action for action in action_path]

        agent = OracleAgent(action_path=grid2op_action_path, action_space=env.action_space,
                            oracle_action_path=action_path,
                            observation_space=None, name=None,
                            init_line_status=init_line_status, init_topo_vect=init_topo_vect)

        # Play OracleAgent and disconnect a line - test if it is well reconnected by agent
        done = False
        for t in range(max_timestep):
            if not done:
                action = agent.act(obs, reward=0., done=False)
                obs, reward, done, info = env.step(action)

        # Check if we obtain the right end topo
        end_topo_vect = obs.topo_vect.tolist()
        expected_end_topo_vect =  [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                                    1,  1,  2,  1,  1,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                                    1, -1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                                    1,  1,  1,  1,  1]
        end_line_status = obs.line_status.tolist()
        expected_end_line_status = [True,  True,  True,  True,  True,  True,  True,  True,  True,
                                    True, False,  True,  True,  True,  True,  True,  True,  True,
                                    True,  True]

        self.assertListEqual(end_topo_vect, expected_end_topo_vect)
        self.assertListEqual(end_line_status, expected_end_line_status)

if __name__ == '__main__':
    unittest.main()
