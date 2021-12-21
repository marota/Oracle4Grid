import time
import unittest

import networkx as nx
import numpy as np

from grid2op.Reward import L2RPNReward
from grid2op.Rules import AlwaysLegal
from oracle4grid.core.actions_utils import combinator
from oracle4grid.core.agent.OracleOverloadReward import OracleOverloadReward
from oracle4grid.core.graph import graph_generator
from oracle4grid.core.graph.attack_graph_module import get_windows_from_df
from oracle4grid.core.graph.graph_generator import get_reachable_topologies
from oracle4grid.core.reward_computation import run_many
from oracle4grid.core.reward_computation.attacks_multiverse import multiverse_simulation, compute_one_multiverse
from oracle4grid.core.reward_computation.run_many import get_node_name, make_df_from_res
from oracle4grid.core.utils.config_ini_utils import MAX_DEPTH, NB_PROCESS, MAX_ITER, REWARD_SIGNIFICANT_DIGIT
from oracle4grid.core.utils.constants import EnvConstants
from oracle4grid.core.utils.launch_utils import OracleParser
from oracle4grid.core.utils.launch_utils import load


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


def init_test_env():
    file = "./oracle4grid/ressources/actions/neurips_track1/ExpertActions_Track1_action_list_score4_reduite.json"
    chronic = "000"
    env_dir = "./data/l2rpn_neurips_2020_track1"
    atomic_actions, env, debug_directory, chronic_id = load(env_dir, chronic, file, False, constants=EnvConstantsTest())
    parser = OracleParser(atomic_actions, env.action_space)
    atomic_actions = parser.parse()

    # 1 - Action generation step
    actions = combinator.generate(atomic_actions, int(config[MAX_DEPTH]), env, False, nb_process=int(config[NB_PROCESS]))

    # 2 - Actions rewards simulation
    reward_df = run_many.run_all(actions, env, int(config[MAX_ITER]), int(config[NB_PROCESS]), debug=False,
                                 agent_seed=16101991, env_seed=16101991)
    return reward_df, env, actions


def with_multiverse():
    reward_df, env, actions = init_test_env()
    # 2.A Adding attacks node, a subgraph for each attack to allow topological actions within an action
    start_time = time.time()
    reward_df, windows = multiverse_simulation(env, actions, reward_df, False, env_seed=16101991, agent_seed=16101991)
    elapsed_time = time.time() - start_time
    print("elapsed_time for attack multiversing is:" + str(elapsed_time))
    return reward_df, env, actions


class GraphValidation(unittest.TestCase):

    def test_fast_forward(self):
        file = "./oracle4grid/ressources/actions/neurips_track1/ExpertActions_Track1_action_list_score4_reduite.json"
        chronic = "000"
        env_dir = "./data/l2rpn_neurips_2020_track1"
        atomic_actions, env, debug_directory,chronic_id = load(env_dir, chronic, file, False, constants=EnvConstantsTest())
        parser = OracleParser(atomic_actions, env.action_space)
        atomic_actions = parser.parse()
        env.chronics_handler.tell_id(0)
        env.seed(0)
        obs = env.reset()
        #0 runner
        obs, reward, done, info = env.step(env.action_space({}))
        obs, reward, done, info = env.step(env.action_space({}))
        obs, reward, done, info = env.step(env.action_space({}))
        obs, reward, done, info = env.step(env.action_space({}))


        env.chronics_handler.tell_id(0)
        env.seed(0)
        obs1 = env.reset()
        # compute the reward when fast forwarding to t = u-1, doing an action that puts us in T at t = u and stopping at v
        env.fast_forward_chronics(3)
        obs1, reward1, done1, info1 = env.step(env.action_space({}))
        assert reward1 == reward

    def test_graph_connected(self):
        reward_df, env, actions = with_multiverse()

        # 3 - Graph generation
        start_time = time.time()
        graph = graph_generator.generate(reward_df, int(config[MAX_ITER]), debug=False, reward_significant_digit=config[REWARD_SIGNIFICANT_DIGIT],
                                         constants=EnvConstantsTest())

        # check first that the full graph is still connected
        g_und = graph.to_undirected()
        assert nx.number_connected_components(g_und) == 1

    def test_graph_duration(self):
        reward_df, env, actions = with_multiverse()

        # 3 - Graph generation
        start_time = time.time()
        graph = graph_generator.generate(reward_df, int(config[MAX_ITER]), debug=False, reward_significant_digit=config[REWARD_SIGNIFICANT_DIGIT],
                                         constants=EnvConstantsTest())
        #Get universal number of connections between nodes for each topo
        topos = get_reachable_topologies(actions)
        transition_per_topo = {topo.name: len(topos[index]) for index, topo in enumerate(actions)}

        all_windows = get_windows_from_df(reward_df)
        # Go through all attacks,
        for window in all_windows:
            start = int(window.split('_')[0])
            end = int(window.split('_')[1])
            nodes_in_all_attacks_inner = []
            for attack in all_windows[window]:
                nodes_in_same_attack = []
                # We loop on all topo to create a list of nodes in the current window of attack
                for topo in all_windows[window][attack]['topos']:
                    topo_name = str(topo)
                    topo_nodes = [get_node_name(topo_name, t, attack) for t in range(start - 2, (end + 1) + 2)]
                    # List needed for first subgraph, and Test1 : we take the attack window, and 2 nodes before and after.
                    nodes_in_same_attack = nodes_in_same_attack + topo_nodes[3:-3]
                    # List needed for second subgraph, and TEst 2 : we zoom on the inner attack window
                    nodes_in_all_attacks_inner = nodes_in_all_attacks_inner + topo_nodes[3:-3]
                subg1 = graph.subgraph(nodes_in_same_attack)
                for t in range(start+2, (end -2)):
                    for topo in all_windows[window][attack]['topos']:
                        topo_name = str(topo)
                        node_name = get_node_name(topo_name, t, attack)
                        # Test n°1 : Check that  there is exactly the same amount of edges in and out of nodes during a attack window
                        nombre_connexion_in = subg1.in_degree(node_name)
                        nombre_connexion_out = subg1.out_degree(node_name)
                        assert nombre_connexion_out == nombre_connexion_in
                        assert nombre_connexion_out == transition_per_topo[topo]

            # Test n°2 : Check that the number of disconected graph in the subgraph containing the attack windows for all topo
            # is the same than the number of attacks. This is needed for all attacks to be disconnected from each others
            subg2 = graph.subgraph(nodes_in_all_attacks_inner)
            subg2 = subg2.to_undirected()
            assert nx.number_connected_components(subg2) == len(all_windows[window])
            # check size of connected components
            if window == '400_447':
                assert [len(c) for c in sorted(nx.connected_components(subg2), key=len, reverse=True)] == [368,368]
        return 1

    def test_multiverse_with_initial_computation(self):
        file = "./oracle4grid/ressources/actions/neurips_track1/ExpertActions_Track1_action_list_score4_reduite.json"
        chronic = "000"
        env_dir = "./data/l2rpn_neurips_2020_track1"
        atomic_actions, env, debug_directory, chronic_id = load(env_dir, chronic, file, False, constants=EnvConstantsTest())
        parser = OracleParser(atomic_actions, env.action_space)
        atomic_actions = parser.parse()

        # 1 - Action generation step
        actions = combinator.generate(atomic_actions, int(config[MAX_DEPTH]), env, False, nb_process=int(config[NB_PROCESS]))

        #Compute one of the runs with runner :
        test_action = actions[1]
        run_runner = run_many.run_one(test_action, env.get_params_for_runner(), int(config[MAX_ITER]), env_seed=[16101991], agent_seed=[16101991])
        #make df and retreive window of attack
        df_runner = make_df_from_res([run_runner], False)
        windows = get_windows_from_df(df_runner)
        # multiverse needs a different environment
        atomic_actions, env, debug_directory, chronic_id = load(env_dir, chronic, file, False, constants=EnvConstantsTest(), opponent_allowed=False)
        #compute same action with multiverse
        for window in windows:
            begin = int(window.split("_")[0])
            end = int(window.split("_")[1])
            attacks = windows[window]
            for attack_id in attacks:
                attack = attacks[attack_id]["attack"]
                run_multiverse = compute_one_multiverse(env, test_action, attack, begin, end, env_seed=16101991, agent_seed=16101991)
                #test expected values for rewards
                expected = run_runner.rewards[begin+1:end+1]
                actual = run_multiverse.rewards
                np.testing.assert_allclose(expected, actual)




if __name__ == '__main__':
    unittest.main()
