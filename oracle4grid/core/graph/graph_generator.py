import math
import warnings
import time

import numpy as np
import pandas as pd
import networkx as nx
from pandas import notnull

from oracle4grid.core.reward_computation.run_many import get_node_name
from oracle4grid.core.utils.constants import END_NODE_REWARD, EnvConstants


def generate(reward_df, max_iter=None, debug=False, reward_significant_digit=None, constants=EnvConstants()):
    if debug:
        print('\n')
        print("============== 3 - Graph generation ==============")

    # Parameters and DataFrame preprocessing
    reward_df = preprocessing(reward_df, max_iter)
    #actions = reward_df['action'].unique()#be careful because with multiprocessing same action name could have different action object, that is diferent pointer adresses
    indice_unique_actions=list(reward_df['name'].drop_duplicates().index)
    actions = reward_df['action'][indice_unique_actions].values

    # Compute possible transitions list for each action
    reachable_topologies_from_init = get_reachable_topologies_from_init(actions, explicit_node_names=debug, constants=constants)

    start_time = time.time()
    reachable_topologies = get_reachable_topologies(actions, explicit_node_names=debug, constants=constants)
    ordered_names = reward_df['name'].unique()

    elapsed_time = time.time() - start_time
    print("get_reachable_topologies time:" + str(elapsed_time))
    # Build graph
    graph = build_transition_graph(reachable_topologies, ordered_names, reward_df, max_iter, reachable_topologies_from_init, reward_significant_digit)
    return graph


def add_nodes_action(graph):
    """
    Add an attribute to the graph representing the action played
    :param graph:
    :return: id of the action played
    """
    dict_ids = {node: (node.split('_t')[0] if node not in ['init', 'end'] else None) for node in graph.nodes}
    nx.set_node_attributes(graph, dict_ids, name='action_id')


def preprocessing(reward_df, max_iter):
    # Maximum timestep to consider for graph
    max_iter_df = reward_df['timestep'].max() + 1
    if max_iter is None:
        max_iter = max_iter_df
    elif max_iter > max_iter_df:
        raise ValueError("max iteration should be <= than simulated timesteps")

    # Filter actions that did not converge until last timestep and raise warning
    simulated_timesteps_per_action = reward_df.groupby('name', as_index=False).agg({'timestep': 'max'})
    divergent_actions = simulated_timesteps_per_action.loc[
        simulated_timesteps_per_action['timestep'] < (max_iter_df - 1), "name"].values
    if len(divergent_actions) > 0:
        reward_df = reward_df[~(reward_df['name'].isin(divergent_actions))]
        warnings.warn("There are " + str(len(
            divergent_actions)) + " actions that have not been simulated until last timestep (diverging simulation)")
    return reward_df


def get_reachable_topologies(actions, explicit_node_names=False, constants=EnvConstants()):
    # Ordered names of actions
    # All possible action transitions
    action_couples = [(action1, action2) for action1 in actions for action2 in actions]
    modified_subs = [action_couple[0].number_of_modified_subs_to(action_couple[1]) for action_couple in
                     action_couples]
    modified_lines = [action_couple[0].number_of_modified_lines_to(action_couple[1]) for action_couple in
                      action_couples]

    # Filter all transitions that violate game rules
    valid_action_couples = [action_couple for action_couple, n_subs, n_lines in zip(action_couples, modified_subs, modified_lines)
                            if (n_subs <= constants.DICT_GAME_PARAMETERS_GRAPH["MAX_SUB_CHANGED"] and n_lines <= constants.DICT_GAME_PARAMETERS_GRAPH[
            "MAX_LINE_STATUS_CHANGED"])]

    # Formattage
    reachable_topologies = []
    if explicit_node_names:
        reachable_topologies = [[action_couple[1].repr for action_couple in valid_action_couples if
                                 action_couple[0].repr == action.repr] for action in actions]
    else:
        reachable_topologies = [[action_couple[1].name for action_couple in valid_action_couples if
                                 action_couple[0].name == action.name] for action in actions]
    return reachable_topologies


def get_reachable_topologies_from_init(actions, explicit_node_names=False, constants=EnvConstants()):
    if not explicit_node_names:
        reachable_topologies_from_init = [action.name
                                          for action in actions
                                          if len(action.subs) <= constants.DICT_GAME_PARAMETERS_GRAPH["MAX_SUB_CHANGED"] and len(action.lines) <=
                                          constants.DICT_GAME_PARAMETERS_GRAPH["MAX_LINE_STATUS_CHANGED"]]
    else:
        reachable_topologies_from_init = [action.repr
                                          for action in actions
                                          if len(action.subs) <= constants.DICT_GAME_PARAMETERS_GRAPH["MAX_SUB_CHANGED"] and len(action.lines) <=
                                          constants.DICT_GAME_PARAMETERS_GRAPH["MAX_LINE_STATUS_CHANGED"]]
    return reachable_topologies_from_init


def create_init_nodes(reachable_topologies_from_init, reward_df):
    ex_init = [str(name) + '_t0' for name in reachable_topologies_from_init]
    or_init = ['init' for i in range(len(reachable_topologies_from_init))]
    weights_init = []
    for action_name in reachable_topologies_from_init:
        reward = reward_df.loc[(reward_df['name'] == action_name) & (reward_df['timestep'] == 0), 'reward'].values[0]
        weights_init.append(reward)
    return or_init, ex_init, weights_init


def create_end_nodes(final_edges, fake_reward=0.1):
    or_end = final_edges
    ex_end = ['end' for i in range(len(final_edges))]
    weights_end = [fake_reward for i in range(len(final_edges))]
    return or_end, ex_end, weights_end


def get_inverted_reachable_topologies(ordered_names, reachable_topologies):
    inverted = []
    for idx, name in enumerate(ordered_names):
        inverted.append([])
        for idxreach, reachable in enumerate(reachable_topologies):
            if name in reachable:
                inverted[idx].append(ordered_names[idxreach])
    return inverted


def build_transition_graph(reachable_topologies, ordered_names, reward_df, max_iter, reachable_topologies_from_init, reward_significant_digit=None):
    """
    Builds a networkx.digraph with all possible transitions and their rewards at each timestep
    :param reachable_topologies: all reachable topologies from all topologies
    :param ordered_names: names of all origin topologies of reachable_topologies, in the same order
    :param reward_df: dataframe with actions and their simulated reward at each timestep. Columns are 'action', 'timestep', reward'
    :param max_iter: maximum simulated timestep
    :return: networkx.digraph with all possible transitions and their rewards at each timestep
    """

    graph_time = time.time()

    # First edges: symbolic init node and reachable first actions
    edges_or, edges_ex, edges_weights = create_init_nodes(reachable_topologies_from_init, reward_df)
    n_init = len(edges_weights)

    # Find and create all possible edges
    start_time = time.time()
    inverted = get_inverted_reachable_topologies(ordered_names, reachable_topologies)
    list_topo = reward_df['name']
    list_attack = reward_df['attack_id']
    list_timestep = reward_df["timestep"]
    list_weights = reward_df["reward"]
    prevAtk = None
    for (topo, attack, t, reward) in zip(list_topo, list_attack, list_timestep, list_weights):
        if t == 0:
            continue
        topo_index = np.where(ordered_names == topo)[0][0]
        for inverted_topo in inverted[topo_index]:
            edges_ex.append(get_node_name(str(topo), t, attack))
            edges_or.append(get_node_name(str(inverted_topo), t - 1, prevAtk))
            edges_weights.append(reward)
        prevAtk = attack

    # Find nodes to dump because impossible to get to from attacks
    uniques_or = set([node for node in edges_or if '_atk' in node])
    uniques_ex = set([node for node in edges_ex if '_atk' in node])
    to_dump = uniques_or.difference(uniques_ex)
    elapsed_time = time.time() - start_time
    print("number of nodes to dump :" + str(len(to_dump)))
    print(elapsed_time)

    # Symbolic end node
    edges_ex_t = [str(name) + '_t' + str(max_iter - 1) for name in ordered_names]
    or_end, ex_end, weights_end = create_end_nodes(edges_ex_t, fake_reward=END_NODE_REWARD)
    edges_or += or_end
    edges_ex += ex_end
    edges_weights += weights_end
    n_end = len(weights_end)

    # Finally create graph object from DataFrame
    edge_df = pd.DataFrame({'or': edges_or, 'ex': edges_ex, 'weight': edges_weights})
    edge_df = edge_df.dropna()

    print("edge_df done - creating graph")
    print("if it takes too long to create, you might change the number of significant digits to consider in config.ini")
    if reward_significant_digit is not None:
        reward_significant_digit = int(reward_significant_digit)
        edge_df.weight = (edge_df.weight * (10 ** (reward_significant_digit))).astype('int64')
        print("currently the number of signficant digits considered is:" + str(reward_significant_digit))
    graph = nx.from_pandas_edgelist(edge_df, target='ex', source='or', edge_attr=['weight'], create_using=nx.DiGraph())

    graph = post_processing_rewards(graph, reward_df)

    # Node removal
    graph = remove_nodes_from_graph_networkx(graph, to_dump)

    print("graph created, total time :")
    elapsed_time = time.time() - graph_time
    print(elapsed_time)
    return graph


def remove_nodes_from_graph_networkx(graph, to_dump):
    '''
    Network x removal of given nodes
    '''
    print("removing nodes created for attack but useless")
    start_time = time.time()
    graph.remove_nodes_from(to_dump)
    elapsed_time = time.time() - start_time
    print("filtering time :")
    print(elapsed_time)
    return graph


def filter_nodes_from_python_list(edges_or, edges_ex, edges_weights, to_dump):
    '''
    Other way of filtering the nodes that should be removed from graph, this is way slower than the node removal from networkx
    '''
    start_time = time.time()
    indexes_to_dump = [i for i in range(len(edges_or)) if edges_or[i] in to_dump]
    edges_ex_filtered = [i for j, i in enumerate(edges_ex) if j not in indexes_to_dump]
    edges_or_filtered = [i for j, i in enumerate(edges_or) if j not in indexes_to_dump]
    edges_weights_filtered = [i for j, i in enumerate(edges_weights) if j not in indexes_to_dump]

    elapsed_time = time.time() - start_time
    print("filtering time :")
    print(elapsed_time)
    return edges_ex_filtered, edges_or_filtered, edges_weights_filtered


def post_processing_rewards(graph, reward_df):
    # Adding other reward
    v_attributes = reward_df[['node_name', 'overload_reward']].set_index('node_name').to_dict('index')
    nx.set_node_attributes(graph, v_attributes)
    return graph
