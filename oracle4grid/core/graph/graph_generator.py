import warnings
import time

import numpy as np
import pandas as pd
import networkx as nx
import itertools

from oracle4grid.core.utils.constants import END_NODE_REWARD, EnvConstants


def generate(reward_df, init_topo_vect, init_line_status, max_iter=None, debug=False, reward_significant_digit=None, constants=EnvConstants()):
    if debug:
        print('\n')
        print("============== 3 - Graph generation ==============")

    actions = reward_df['action'].unique()
    # Parameters and DataFrame preprocessing
    reward_df = preprocessing(reward_df, max_iter, explicit_node_names=debug)

    # Compute possible transitions list for each action
    reachable_topologies_from_init = get_reachable_topologies_from_init(actions, init_topo_vect, init_line_status,
                                                                        explicit_node_names=debug, constants=constants)

    start_time = time.time()
    reachable_topologies = get_reachable_topologies(actions, init_topo_vect, init_line_status,
                                                    explicit_node_names=debug, constants=constants)
    ordered_names = reward_df['name'].unique()

    elapsed_time = time.time() - start_time
    print("get_reachable_topologies time:" + str(elapsed_time))
    # Build graph
    graph = build_transition_graph(reachable_topologies, ordered_names, reward_df, max_iter, reachable_topologies_from_init, reward_significant_digit)
    # graph = add_nodes_action(graph)
    return graph


def add_nodes_action(graph):
    """
    Add an attribute to the graph representing the action played
    :param graph:
    :return: id of the action played
    """
    dict_ids = {node: (node.split('_t')[0] if node not in ['init', 'end'] else None) for node in graph.nodes}
    nx.set_node_attributes(graph, dict_ids, name='action_id')


def preprocessing(reward_df, max_iter, explicit_node_names=False):
    # Extract name column
    if not explicit_node_names:
        reward_df['name'] = [action.name for action in reward_df['action']]
    else:
        reward_df['name'] = [str(action) for action in reward_df['action']]

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


def get_reachable_topologies(actions, init_topo_vect, init_line_status, explicit_node_names=False, constants=EnvConstants()):
    # Ordered names of actions
    # ordered_names = reward_df['name'].unique()
    # TODO: explicit node names if asked

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


def get_reachable_topologies_from_init(actions, init_topo_vect, init_line_status, explicit_node_names=False, constants=EnvConstants()):
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

    ## Reachable transitions for each timestep and associated rewards
    start_time = time.time()
    edges_or += [str(ordered_names[i]) + '_t' + str(int(t))
                 for i in range(len(reachable_topologies))
                 for reachable_topo in reachable_topologies[i]
                 for t in range(max_iter - 1)]
    edges_ex += [str(reachable_topo) + '_t' + str(int(t + 1))
                 for i in range(len(reachable_topologies))
                 for reachable_topo in reachable_topologies[i]
                 for t in range(max_iter - 1)]

    reward_table = reward_df[['timestep', 'reward', 'name']].pivot(index='timestep', columns='name', values='reward')

    # edges_weights += [reward_table[reachable_topo][int(t + 1)] for i in range(len(reachable_topologies))
    #                  for reachable_topo in reachable_topologies[i]
    #                  for t in range(max_iter - 1)]
    edges_weights += list(itertools.chain(*[reward_table[reachable_topologies[i]].loc[1:max_iter].values.flatten(order='F')
                                            for i in range(len(reachable_topologies))]))
    elapsed_time = time.time() - start_time
    print(elapsed_time)

    # Symbolic end node
    edges_ex_t = [str(name) + '_t' + str(max_iter - 1) for name in ordered_names]  # edges_ex.copy()
    or_end, ex_end, weights_end = create_end_nodes(edges_ex_t, fake_reward=END_NODE_REWARD)
    edges_or += or_end
    edges_ex += ex_end
    edges_weights += weights_end

    # Finally create graph object from DataFrame
    edge_df = pd.DataFrame({'or': edges_or, 'ex': edges_ex, 'weight': edges_weights})
    edge_df = edge_df.dropna()

    print("edge_df done - creating graph")
    print("if it takes too long to create, you might change the number of significant digits to consider in config.ini")
    if (reward_significant_digit is not None):
        reward_significant_digit = int(reward_significant_digit)
        edge_df.weight = (edge_df.weight * (10 ** (reward_significant_digit))).astype('int64')
        print("currently the number of signficant digits considered is:" + str(reward_significant_digit))
    graph = nx.from_pandas_edgelist(edge_df, target='ex', source='or', edge_attr=['weight'], create_using=nx.DiGraph())
    graph = post_processing_rewards(graph, reward_df)
    print("graph created")
    return graph


def post_processing_rewards(graph, reward_df):
    for node in graph.nodes:
        if node is "init" or node is "end":
            continue
        action = node.split("_t")[0]
        timestep = node.split("_t")[1]
        line = reward_df.loc[(reward_df['name'].astype(str) == str(action)) & (reward_df['timestep'] == int(timestep))]
        graph.nodes[node]["overload_reward"] = line.iloc[0]["overload_reward"]
    return graph

