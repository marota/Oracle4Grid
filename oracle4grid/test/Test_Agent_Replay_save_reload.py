import os.path
import unittest

import numpy as np
import pandas as pd
from grid2op.Episode import EpisodeData
from grid2op.dtypes import dt_float, dt_bool
from oracle4grid.core.utils.prepare_environment import prepare_env, get_initial_configuration
from oracle4grid.core.actions_utils import combinator
from oracle4grid.core.utils.launch_utils import OracleParser,load
from grid2op.Agent import DoNothingAgent # for example...
from grid2op.Runner import Runner
from oracle4grid.core.replay import agent_replay
from oracle4grid.core.agent.OracleAgent import OracleAgent
from oracle4grid.core.oracle import oracle, save_oracle_data_for_replay, load_oracle_data_for_replay
import json
from oracle4grid.core.utils.constants import EnvConstants
from grid2op.Parameters import Parameters
import configparser
from grid2op.Reward import L2RPNReward
from oracle4grid.core.agent.OracleOverloadReward import OracleOverloadReward
from grid2op.Rules import AlwaysLegal


BEST_PATH_NAME = "Best possible path with game rules"
INDICATORS_NAMES_COL = "Indicator name"
INDICATORS_REWARD_COL = "Reward value"

CONFIG = {
    "max_depth": 2,
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

class PerformanceTest(unittest.TestCase):
    def test_save_reload_oracle_actions(self):
        action_file = "./oracle4grid/test_resourses/test_unitary_actions.json"
        chronic = "000"
        env_dir = "./data/rte_case14_realistic"
        path_save = 'oracle4grid/output/rte_case14_realistic/scenario_0/replay_test'

        env_seed = 16101991
        agent_seed = 16101991

        param = Parameters()
        constants = EnvConstants()
        param.init_from_dict(constants.DICT_GAME_PARAMETERS_SIMULATION)


        # Load unitary actions
        # Compute a fake Oracle action path and Oracle Agent
        atomic_actions, env, debug_directory, chronic_id = load(env_dir, chronic, action_file, debug=False,
                                                                         constants=EnvConstantsTest())

        res = oracle(atomic_actions, env, False, config=CONFIG, debug_directory=None,
                     agent_seed=agent_seed, env_seed=env_seed,
                     grid_path=env_dir, chronic_scenario=chronic_id, constants=constants)#

        best_path, grid2op_action_path, best_path_no_overload, grid2op_action_path_no_overload, kpis = res

        grid2op_action_list = grid2op_action_path
        oracle_action_list = best_path
        save_oracle_data_for_replay(oracle_action_list, path_save)

        action_list_reloaded, init_topo_vect, init_line_status, oracle_actions_in_path=load_oracle_data_for_replay(env,action_file,path_save,action_depth=CONFIG["max_depth"])
        assert([action_list_reloaded[i]==grid2op_action_list[i] for i in range(len(action_list_reloaded))])
        assert ([oracle_actions_in_path[i].repr == oracle_action_list[i].repr for i in range(len(oracle_action_list))])

    def test_oracle_agent_with_action_path_reload(self):
        action_file = "./oracle4grid/test_resourses/test_unitary_actions.json"
        chronic = "000"
        env_dir = "./data/rte_case14_realistic"
        path_save = 'oracle4grid/output/rte_case14_realistic/scenario_0/replay_test'

        env_seed = 16101991
        agent_seed = 16101991

        param = Parameters()
        constants = EnvConstants()
        param.init_from_dict(constants.DICT_GAME_PARAMETERS_SIMULATION)


        # Load unitary actions
        # Compute a fake Oracle action path and Oracle Agent
        atomic_actions, env, debug_directory, chronic_id = load(env_dir, chronic, action_file, debug=False,
                                                                         constants=EnvConstantsTest())
        
        action_list_reloaded, init_topo_vect, init_line_status, oracle_actions_in_path = load_oracle_data_for_replay(
            env, action_file, path_save, action_depth=CONFIG["max_depth"])
        
        agent = OracleAgent(env.action_space, action_path=action_list_reloaded,
                            oracle_action_path=oracle_actions_in_path,
                            init_topo_vect=init_topo_vect, init_line_status=init_line_status)  # .gen_next(action_path)

        runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=agent)
        scenario_name,rewrad,timesteps,episode=runner.run_one_episode(indx=chronic_id,
                                               path_save=path_save,
                                               pbar=True,
                                               env_seed=env_seed,  # ENV_SEEDS,
                                               max_iter=CONFIG["max_iter"],
                                               agent_seed=agent_seed,
                                               # AGENT_SEEDS,
                                               detailed_output=True)

        #check that actions from initial action path and after replay are the same
        assert(all([all(episode.actions[i].to_vect()==action_list_reloaded[i].to_vect()) for i in range(len(action_list_reloaded))]))