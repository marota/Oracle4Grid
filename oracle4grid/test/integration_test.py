import unittest
from oracle4grid.main import load_and_run
from oracle4grid.core.agent.OracleAgent import OracleAgent
from oracle4grid.core.utils.prepare_environment import prepare_game_params, prepare_env


BEST_PATH_NAME = "Best possible path with game rules"
INDICATORS_NAMES_COL = "Indicator name"
INDICATORS_REWARD_COL = "Reward value"

CONFIG = {
    "max_depth": 3,
    "max_iter": 6,
    "nb_process": 1,
    "best_path_type": "shortest",
    "n_best_topos" : 2
}


class IntegrationTest(unittest.TestCase):
    def test_base_run(self):
        file = "./oracle4grid/ressources/actions/test_unitary_actions.json"
        chronic = "000"
        env_dir = "./oracle4grid/ressources/grids/rte_case14_realistic"
        action_path, grid2op_action_path = load_and_run(env_dir, chronic, file, False, CONFIG)
        self.assertNotEqual(action_path, None)
        return 1

    def test_best_path_equal(self):
        file = "./oracle4grid/ressources/actions/test_unitary_actions.json"
        chronic = "000"
        env_dir = "./oracle4grid/ressources/grids/rte_case14_realistic"
        action_path, grid2op_action_path, indicators= load_and_run(env_dir, chronic, file, False, CONFIG)
        best_path_actual = list(map(lambda x : x.__str__(), action_path))
        best_path_expected = ['sub-5-2', 'sub-1-1_sub-5-2', 'sub-1-1_sub-5-2', 'sub-1-1_sub-5-2', 'sub-1-1_sub-5-2', 'sub-1-1_sub-5-2']
        self.assertListEqual(best_path_actual, best_path_expected)

    def test_agent_rewards(self):
        # Compute best path with Oracle
        file = "./oracle4grid/ressources/actions/test_unitary_actions.json"
        chronic = "000"
        env_dir = "./oracle4grid/ressources/grids/rte_case14_realistic"
        action_path, grid2op_action_path, indicators = load_and_run(env_dir, chronic, file, False, CONFIG)
        best_path_reward = float(indicators.loc[indicators[INDICATORS_NAMES_COL]==BEST_PATH_NAME,INDICATORS_REWARD_COL].values[0])

        # Replay path with OracleAgent as standard gym episode replay (OracleAgent not compatible with Grid2op Runner yet)
        param = prepare_game_params()
        env = prepare_env(env_dir, chronic, param)
        agent = OracleAgent(action_path=grid2op_action_path, action_space=env.action_space,
                            observation_space=None, name=None)
        agent_reward = 0.
        done = False
        obs = env.reset()
        for t in range(CONFIG['max_iter']):
            if not done:
                action = agent.act(obs, reward=0., done=False)
                obs, reward, done, info = env.step(action)
                agent_reward += reward

        # TODO: actions are correct (at least the first one) but impossible to get same reward as Runner(reward_df)...
        # TODO: Careful with 0.1 at the end of scenario?

        # Check if we get expected reward
        self.assertEqual(best_path_reward, agent_reward)


if __name__ == '__main__':
    unittest.main()
