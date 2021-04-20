import math
import warnings
import time

import numpy as np
import pandas as pd
import networkx as nx
import itertools

from oracle4grid.core.graph import attack_graph_module
from oracle4grid.core.reward_computation.run_many import get_node_name
from oracle4grid.core.utils.constants import END_NODE_REWARD, EnvConstants


def generate(reward_df, max_iter=None, debug=False, reward_significant_digit=None, constants=EnvConstants()):
    if debug:
        print('\n')
        print("============== 3 - Graph generation ==============")

    # Parameters and DataFrame preprocessing
    reward_df = preprocessing(reward_df, max_iter)
    actions = reward_df['action'].unique()

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


def create_edges_at_t(edges_ex_previous, ordered_names, reachable_topologies, t):
    """
    From extremities of previous timestep, compute extremities of next timestep
    :param edges_ex_previous:
    :param ordered_names:
    :param reachable_topologies:
    :param t:
    :return:
    """
    old_suffix = '_t' + str(t)
    new_suffix = '_t' + str(t + 1)
    edges_or_t = []
    edges_ex_t = []
    for or_ in edges_ex_previous:
        # Find possible transitions of each origin node
        try:
            or_name = int(or_.replace(old_suffix, ''))
        except:
            or_name = or_.replace(old_suffix, '')
        or_position = np.where(ordered_names == or_name)[0][0]
        new_edges_ex = [str(ex_name) + new_suffix for ex_name in reachable_topologies[or_position]]
        # Append to edges
        edges_ex_t += new_edges_ex
        edges_or_t += [or_ for ex_ in new_edges_ex]
    return edges_or_t, edges_ex_t


def get_transition_rewards_from_t(reward_df, edges_ex_t, t):
    """
    This function return the weights of the edges of the graph that connect t to t+1
    Between timestep t and t+1, the edges are weighted by the reward obtained in t+1 (by applying the action represented by the edge extremity)
    :param reward_df: dataframe with actions and their simulated reward at each timestep. Columns are 'action', 'timestep', reward'
    :param edges_ex_t: names of the actions that generate the rewards at t+1
    :param t: origin timestep of the edge
    :return: list of rewards at t+1 which
    """
    edges_weights_t = []
    for action_name in edges_ex_t:
        try:
            action_name = int(action_name.replace("_t" + str(t + 1), ""))
        except:
            action_name = action_name.replace("_t" + str(t + 1), "")
        reward = reward_df.loc[(reward_df['name'] == action_name) & (reward_df['timestep'] == (t + 1)), 'reward'].values[0]
        edges_weights_t.append(reward)
    return edges_weights_t


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

    # First edges: symbolic init node and reachable first actions
    edges_or, edges_ex, edges_weights = create_init_nodes(reachable_topologies_from_init, reward_df)
    n_init = len(edges_weights)

    # Optional old way
    """
    edges_ex_old = edges_ex.copy()
    edges_or_old = edges_or.copy()
    edges_weights_old = edges_weights.copy()
    edges_or_old += [str(ordered_names[i]) + '_t' + str(int(t))
                 for i in range(len(reachable_topologies))
                 for reachable_topo in reachable_topologies[i]
                 for t in range(max_iter - 1)]
    edges_ex_old += [str(reachable_topo) + '_t' + str(int(t + 1))
                 for i in range(len(reachable_topologies))
                 for reachable_topo in reachable_topologies[i]
                 for t in range(max_iter - 1)]
    reward_table = reward_df[['timestep', 'reward', 'name']].pivot(index='timestep', columns='name', values='reward')

    edges_weights_old += list(itertools.chain(*[reward_table[reachable_topologies[i]].loc[1:max_iter].values.flatten(order='F')
                                            for i in range(len(reachable_topologies))]))
    """

    # Reachable transitions for each timestep and associated rewards
    start_time = time.time()
    inverted = get_inverted_reachable_topologies(ordered_names, reachable_topologies)
    reward_df_with_edges = reward_df
    list_topo = reward_df_with_edges['name']
    list_attack = reward_df_with_edges['attack_id']
    list_timestep = reward_df_with_edges["timestep"]
    list_weights = reward_df_with_edges["reward"]

    for (topo, attack, t, reward) in zip(list_topo, list_attack, list_timestep, list_weights):
        if t == 0:
            continue
        topo_index = np.where(ordered_names == topo)[0][0]
        for inverted_topo in inverted[topo_index]:
            edges_ex.append(get_node_name(str(topo), t, attack))
            edges_or.append(get_node_name(str(inverted_topo), t - 1, attack))
            edges_weights.append(reward)

    uniques_or = set([node for node in edges_or if '_atk' in node])
    uniques_ex = set([node for node in edges_ex if '_atk' in node])
    to_dump = uniques_or.difference(uniques_ex)
    print("number of nodes to dump :" + str(len(to_dump)))

    # Optional filtering
    indexes_to_dump = [i for i in range(len(edges_or)) if edges_or[i] in to_dump]
    edges_ex_filtered = [i for j, i in enumerate(edges_ex) if j not in indexes_to_dump]
    edges_or_filtered = [i for j, i in enumerate(edges_or) if j not in indexes_to_dump]
    edges_weights_filtered = [i for j, i in enumerate(edges_weights) if j not in indexes_to_dump]

    # Optional dataframe generation
    """
    df1 = pd.DataFrame(columns=["ex", "or", "weight"])
    df1["ex"] = edges_ex
    df1["or"] = edges_or
    df1["weight"] = edges_weights
    df2 = pd.DataFrame(columns=["ex", "or", "weight"])
    df2["ex"] = edges_ex_old
    df2["or"] = edges_or_old
    df2["weight"] = edges_weights_old
    """

    elapsed_time = time.time() - start_time
    print(elapsed_time)

    # Symbolic end node
    edges_ex_t = [str(name) + '_t' + str(max_iter - 1) for name in ordered_names]
    or_end, ex_end, weights_end = create_end_nodes(edges_ex_t, fake_reward=END_NODE_REWARD)
    edges_or_filtered += or_end
    edges_ex_filtered += ex_end
    edges_weights_filtered += weights_end
    n_end = len(weights_end)

    # Removing attack edges
    """
    print("Removing attack edges")
    start_time = time.time()
    edges_or, edges_ex, edges_weights = attack_graph_module.filter_attacked_nodes(edges_or, edges_ex, edges_weights, reward_df)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    """
    ### other_way to remove inconsistent attack edges after graph creation, using graph.remove_edges then. Important to do it before edge_df.dropna()
    # attack_filters=attack_graph_module.filter_attack_edges(reachable_topologies,ordered_names,reward_df,max_iter,n_init,n_end)
    # edge_df = pd.DataFrame({'or': edges_or, 'ex': edges_ex, 'weight': edges_weights,'attack_filter':attack_filters})
    ##

    # Finally create graph object from DataFrame

    edge_df = pd.DataFrame({'or': edges_or_filtered, 'ex': edges_ex_filtered, 'weight': edges_weights_filtered})
    edge_df = edge_df.dropna()

    print("edge_df done - creating graph")
    print("if it takes too long to create, you might change the number of significant digits to consider in config.ini")
    if (reward_significant_digit is not None):
        reward_significant_digit = int(reward_significant_digit)
        edge_df.weight = (edge_df.weight * (10 ** (reward_significant_digit))).astype('int64')
        print("currently the number of signficant digits considered is:" + str(reward_significant_digit))
    graph = nx.from_pandas_edgelist(edge_df, target='ex', source='or', edge_attr=['weight'], create_using=nx.DiGraph())

    # other_way to remove inconsistent attack edges after graph creation
    # graph = nx.from_pandas_edgelist(edge_df, target='ex', source='or', edge_attr=['weight', 'attack_filter'],create_using=nx.DiGraph())
    # graph.remove_edges_from([e for i,e in enumerate(graph.edges(data=True)) if e[2]['attack_filter']])
    ###

    graph = post_processing_rewards(graph, reward_df)

    print("removing nodes created for attack but useless")
    graph.remove_nodes_from(to_dump)
    print("graph created")

    return graph


def post_processing_rewards(graph, reward_df):
    # Adding other reward
    v_attributes = reward_df[['node_name', 'overload_reward']].set_index('node_name').to_dict('index')
    nx.set_node_attributes(graph, v_attributes)
    return graph


def attack_filter(row):
    keep = False
    for att in row["attack"]:
        if len(att.as_dict) != 0:
            keep = True
    return keep
