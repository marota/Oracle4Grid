import time
import unittest
import warnings
import json
from oracle4grid.core.graph.attack_graph_module import get_windows_from_df

import networkx as nx

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
    "max_iter": 450,
    "nb_process": 1,
    "best_path_type": "longest",
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
        file = "./oracle4grid/ressources/actions/neurips_track1/ExpertActions_Track1_action_list_score4_reduite.json"
        chronic = "000"
        env_dir = "./data/l2rpn_neurips_2020_track1"
        atomic_actions, env, debug_directory = load(env_dir, chronic, file, False, constants=EnvConstantsTest())
        parser = OracleParser(atomic_actions, env.action_space)
        atomic_actions = parser.parse()
        # 0 - Preparation : Get initial topo and line status
        #init_topo_vect, init_line_status = get_initial_configuration(env)

        # 1 - Action generation step
        start_time = time.time()
        actions = combinator.generate(atomic_actions, int(config[MAX_DEPTH]), env, False, nb_process=int(config[NB_PROCESS]))
        elapsed_time = time.time() - start_time
        print("elapsed_time for action generation is:" + str(elapsed_time))

        # 2 - Actions rewards simulation
        start_time = time.time()
        reward_df = run_many.run_all(actions, env, int(config[MAX_ITER]), int(config[NB_PROCESS]), debug=False,
                                     agent_seed=[16101991], env_seed=[16101991])

        # 3 - Graph generation
        start_time = time.time()
        graph = graph_generator.generate(reward_df, int(config[MAX_ITER]), debug=False, reward_significant_digit=config[REWARD_SIGNIFICANT_DIGIT], constants=EnvConstantsTest())

        #check first that the full graph is still connected
        g_und = graph.to_undirected()
        assert nx.number_connected_components(g_und) == 1

        all_windows = get_windows_from_df(reward_df)
        # Go through all attacks,
        for window in all_windows:
            start = int(window.split('_')[0])
            end = int(window.split('_')[1])
            nodes_in_all_attacks_inner = []
            for attack in all_windows[window]:
                nodes_in_same_attack = []
                # We loop on all topo to create a list of nodes in the current window of attack
                for topo in all_windows[window][attack]:
                    topo_name = str(topo)
                    topo_nodes = [topo_name + '_t' + str(t) for t in range(start - 2, (end + 1) + 2)]
                    # List needed for first subgraph, and Test1 : we take the attack window, and 2 nodes before and after.
                    nodes_in_same_attack = nodes_in_same_attack + topo_nodes
                    # List needed for second subgraph, and TEst 2 : we zoom on the inner attack window
                    nodes_in_all_attacks_inner = nodes_in_all_attacks_inner + topo_nodes[3:-3]
                subg1 = graph.subgraph(nodes_in_same_attack)
                for t in range(start - 1, (end + 1) + 1):
                    for topo in all_windows[window][attack]:
                        topo_name = str(topo)
                        # Test n°1 : Check that  there is exactly the same amount of edges in and out of nodes during a attack window
                        nombre_connexion_in = subg1.in_degree(topo_name + '_t' + str(t))
                        nombre_connexion_out = subg1.out_degree(topo_name + '_t' + str(t))
                        assert nombre_connexion_out == nombre_connexion_in

            # Test n°2 : Check that the number of disconected graph in the subgraph containing the attack windows for all topo
            # is the same than the number of attacks. This is needed for all attacks to be disconnected from each others
            subg2 = graph.subgraph(nodes_in_all_attacks_inner)
            subg2 = subg2.to_undirected()
            assert nx.number_connected_components(subg2) == len(all_windows[window])
            #check size of connected components
            if window=='400_447':
                assert [len(c) for c in sorted(nx.connected_components(subg2), key=len, reverse=True)]==[184, 184]
        return 1


if __name__ == '__main__':
    unittest.main()
