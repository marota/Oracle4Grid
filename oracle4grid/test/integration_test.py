import unittest

from oracle4grid.core.actions_utils import combinator
from oracle4grid.core.reward_computation import run_many
from oracle4grid.core.utils.config_ini_utils import MAX_DEPTH, NB_PROCESS, MAX_ITER
from oracle4grid.core.utils.launch_utils import load_and_run, load
from oracle4grid.core.agent.OracleAgent import OracleAgent
from oracle4grid.core.utils.prepare_environment import prepare_game_params, prepare_env, get_initial_configuration

from pandas.testing import assert_frame_equal
import pandas as pd
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


class IntegrationTest(unittest.TestCase):
    def test_base_run(self):
        file = "./oracle4grid/ressources/actions/rte_case14_realistic/test_unitary_actions.json"
        chronic = "000"
        env_dir = "./data/rte_case14_realistic"
        action_path, grid2op_action_path, kpis = load_and_run(env_dir, chronic, file, False,None,None, CONFIG)
        self.assertNotEqual(action_path, None)
        return 1

    def test_best_path_equal(self):
        file = "./oracle4grid/ressources/actions/rte_case14_realistic/test_unitary_actions.json"
        chronic = "000"
        env_dir = "./data/rte_case14_realistic"
        action_path, grid2op_action_path, indicators = load_and_run(env_dir, chronic, file, False,None,None, CONFIG)
        best_path_actual = list(map(lambda x: x.__str__(), action_path))
        best_path_expected = ['sub-5-2', 'sub-1-1_sub-5-2', 'sub-1-1_sub-5-2', 'sub-1-1_sub-5-2', 'sub-1-1_sub-5-2', 'sub-1-1_sub-5-2']
        self.assertListEqual(best_path_actual, best_path_expected)

    def test_agent_rewards(self):
        # Compute best path with Oracle
        file = "./oracle4grid/ressources/actions/rte_case14_realistic/test_unitary_actions.json"
        chronic = 0
        env_dir = "./data/rte_case14_realistic"
        action_path, grid2op_action_path, indicators = load_and_run(env_dir, chronic, file, False,None,None, CONFIG)
        best_path_reward = float(indicators.loc[indicators[INDICATORS_NAMES_COL] == BEST_PATH_NAME, INDICATORS_REWARD_COL].values[0])

        # Replay path with OracleAgent as standard gym episode replay (OracleAgent not compatible with Grid2op Runner yet)
        param = prepare_game_params()
        env = prepare_env(env_dir, chronic, param)
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
        atomic_actions, env, debug_directory = load(env_dir, chronic, file, False)
        # 0 - Preparation : Get initial topo and line status
        init_topo_vect, init_line_status = get_initial_configuration(env)

        # 1 - Action generation step
        actions = combinator.generate(atomic_actions, int(CONFIG[MAX_DEPTH]), env, False, init_topo_vect, init_line_status)
        actions = map(lambda x: str(x), actions)
        self.assertListEqual(list(actions), self.expected_actions)

    def test_kpi(self):
        file = "./oracle4grid/ressources/actions/rte_case14_realistic/test_unitary_actions.json"
        chronic = "000"
        env_dir = "./data/rte_case14_realistic"
        action_path, grid2op_action_path, indicators = load_and_run(env_dir, chronic, file, False,None,None, CONFIG)
        expected = pd.read_csv('./oracle4grid/test_resourses/test_kpi.csv', sep=',', index_col=0)
        assert_frame_equal(indicators, expected)

    def test_reward_df(self):
        file = "./oracle4grid/ressources/actions/rte_case14_realistic/test_unitary_actions.json"
        chronic = "000"
        env_dir = "./data/rte_case14_realistic"
        atomic_actions, env, debug_directory = load(env_dir, chronic, file, False)
        # 0 - Preparation : Get initial topo and line status
        init_topo_vect, init_line_status = get_initial_configuration(env)

        # 1 - Action generation step
        actions = combinator.generate(atomic_actions, int(CONFIG[MAX_DEPTH]), env, False, init_topo_vect, init_line_status)
        # 2 - Actions rewards simulation
        reward_df = run_many.run_all(actions, env, int(CONFIG[MAX_ITER]), int(CONFIG[NB_PROCESS]), debug=False)
        reward_df["action"] = reward_df["action"].astype(str)
        expected = pd.read_csv('./oracle4grid/test_resourses/test_reward.csv', sep=',', index_col=0)
        assert_frame_equal(reward_df, expected)


if __name__ == '__main__':
    unittest.main()
