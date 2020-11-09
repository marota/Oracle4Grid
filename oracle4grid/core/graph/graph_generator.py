import numpy as np
import pandas as pd
import networkx as nx

from oracle4grid.core.utils.constants import DICT_GAME_PARAMETERS_GRAPH


def generate(reward_df, max_depth, init_topo_vect, init_line_status, debug = False):
    # Compute possible transitions list for each action
    reward_df['name'] = [action.name for action in reward_df['action']]
    reachable_topologies_from_init = get_reachable_topologies_from_init(reward_df, init_topo_vect, init_line_status,
                                                                        explicit_node_names=debug)
    reachable_topologies, ordered_names = get_reachable_topologies(reward_df, init_topo_vect, init_line_status,
                                                                   explicit_node_names=debug)

    # Build graph
    graph = build_transition_graph(reachable_topologies, ordered_names, reward_df, max_depth, reachable_topologies_from_init)
    return graph


def get_reachable_topologies(reward_df, init_topo_vect, init_line_status, explicit_node_names = False):
    # Ordered names of actions
    ordered_names = reward_df['name'].unique()
    # TODO: explicit node names if asked

    # All possible action transitions
    actions = reward_df['action'].unique()
    action_couples = [(action1, action2) for action1 in actions for action2 in actions]
    modified_subs = [len(action_couple[0].modified_subs_to(action_couple[1], init_topo_vect))
                     for action_couple in action_couples]
    modified_lines = [len(action_couple[0].modified_lines_to(action_couple[1], init_line_status))
                      for action_couple in action_couples]

    # Filter all transitions that violate game rules
    valid_action_couples = [action_couple for action_couple, n_subs, n_lines in zip(action_couples, modified_subs, modified_lines)
                            if (n_subs <= DICT_GAME_PARAMETERS_GRAPH["MAX_SUB_CHANGED"] and n_lines <= DICT_GAME_PARAMETERS_GRAPH["MAX_LINE_STATUS_CHANGED"])]

    # Formattage
    reachable_topologies = []
    for action in actions:
        reachable_topologies_from_action = [action_couple[1].name for action_couple in valid_action_couples if action_couple[0].name == action.name]
        reachable_topologies.append(reachable_topologies_from_action)
    return reachable_topologies, ordered_names

def get_reachable_topologies_from_init(reward_df, init_topo_vect, init_line_status, explicit_node_names = False):
    actions = reward_df['action'].unique()
    if not explicit_node_names:
        reachable_topologies_from_init = [action.name
                                          for action in actions
                                          if len(action.subs) <= DICT_GAME_PARAMETERS_GRAPH["MAX_SUB_CHANGED"] and len(action.lines) <= DICT_GAME_PARAMETERS_GRAPH["MAX_LINE_STATUS_CHANGED"]]
    # TODO: else: pareil avec fonction qui crée un nom explicite
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
    old_suffix = '_t'+str(t)
    new_suffix = '_t'+str(t+1)
    edges_or_t = []
    edges_ex_t = []
    for or_ in edges_ex_previous:
        # Find possible transitions of each origin node
        or_name = int(or_.replace(old_suffix, ''))
        or_position = np.where(ordered_names==or_name)[0][0]
        new_edges_ex = [str(ex_name)+new_suffix for ex_name in reachable_topologies[or_position]]
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
        action_name = int(action_name.replace("_t"+str(t+1), ""))
        reward = reward_df.loc[(reward_df['name']==action_name)&(reward_df['timestep']==(t+1)), 'reward'].values[0]
        edges_weights_t.append(reward)
    return edges_weights_t

def create_init_nodes(reachable_topologies_from_init, reward_df):
    ex_init = [str(name) +'_t0' for name in reachable_topologies_from_init]
    or_init = ['init' for i in range(len(reachable_topologies_from_init))]
    weights_init = []
    for action_name in reachable_topologies_from_init:
        reward = reward_df.loc[(reward_df['name'] == action_name) & (reward_df['timestep'] == 0), 'reward'].values[0]
        weights_init.append(reward)
    return or_init, ex_init, weights_init

def create_end_nodes(final_edges, fake_reward = 0.1):
    or_end = final_edges
    ex_end = ['end' for i in range(len(final_edges))]
    weights_end = [fake_reward for i in range(len(final_edges))]
    return or_end, ex_end, weights_end


def build_transition_graph(reachable_topologies, ordered_names, reward_df, max_iter, reachable_topologies_from_init):
    """
    Builds a networkx.digraph with all possible transitions and their rewards at each timestep
    :param reachable_topologies: all reachable topologies from all topologies
    :param ordered_names: names of all origin topologies of reachable_topologies, in the same order
    :param reward_df: dataframe with actions and their simulated reward at each timestep. Columns are 'action', 'timestep', reward'
    :param max_iter: maximum simulated timestep
    :return: networkx.digraph with all possible transitions and their rewards at each timestep
    """
    # or_structure, ex_structure = create_edge_structure(reachable_topologies, ordered_names)

    # First edges: symbolic init node and reachable first actions
    edges_or, edges_ex, edges_weights = create_init_nodes(reachable_topologies_from_init, reward_df)

    # Initialize
    edges_ex_t = edges_ex.copy()

    # Reachable transitions for each timestep and associated rewards
    for t in range(max_iter-1):
        # Extremities should be treated once each
        edges_ex_t = pd.unique(edges_ex_t).tolist()

        # Compute edges and their weights. These edges connect timesteps t and t+1 with all possible ways
        edges_or_t, edges_ex_t = create_edges_at_t(edges_ex_t, ordered_names, reachable_topologies, t)
        edges_weights_t = get_transition_rewards_from_t(reward_df, edges_ex_t, t)

        # Append to other edges from other timesteps
        edges_or += edges_or_t.copy()
        edges_ex += edges_ex_t.copy()
        edges_weights += edges_weights_t.copy()

    # Symbolic end node
    or_end, ex_end, weights_end = create_end_nodes(edges_ex_t)
    edges_or += or_end
    edges_ex += ex_end
    edges_weights += weights_end

    # Finally create graph object from DataFrame
    edge_df = pd.DataFrame({'or': edges_or, 'ex': edges_ex, 'weight': edges_weights})
    graph = nx.from_pandas_edgelist(edge_df, target='ex', source='or', edge_attr=['weight'], create_using=nx.DiGraph())
    return graph