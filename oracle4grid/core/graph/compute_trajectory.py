# Use algo to find best path
import networkx as nx
import numpy as np


SHORTEST = "shortest"
LONGEST = "longest"


def best_path(graph, best_path_type, actions, init_topo_vect, init_line_status, debug = False):
    if debug:
        print('\n')
        print("============== 4 - Computation of best action path ==============")
    path = None
    if best_path_type == SHORTEST:
        path = nx.bellman_ford_path(graph, 'init', 'end')  # shortest_path(G)
    elif best_path_type == LONGEST:
        path = nx.dag_longest_path(graph)  # Longest path
    # Add more treatment here

    # Traduce path in terms of OracleActions
    action_path = return_action_path(path, actions, debug=debug)
    grid2op_action_path = get_grid2op_action_path(action_path, init_topo_vect, init_line_status)
    return action_path, grid2op_action_path


def best_path_no_overload(graph, best_path_type, actions, init_topo_vect, init_line_status, debug = False):
    if debug:
        print('\n')
        print("============== 4 - Computation of best action path with no overload ==============")

    path = None
    graph_no_overload = graph.copy()
    nodes = graph_no_overload.nodes()
    for node in nodes:
        if node.endswith("_overload"):
            graph_no_overload.remove(node)
    if best_path_type == SHORTEST:
        path = nx.bellman_ford_path(graph_no_overload, 'init', 'end')  # shortest_path(G)
    elif best_path_type == LONGEST:
        path = nx.dag_longest_path(graph_no_overload)  # Longest path
    # Add more treatment here

    # Traduce path in terms of OracleActions
    action_path = return_action_path(path, actions, debug=debug)
    grid2op_action_path = get_grid2op_action_path(action_path, init_topo_vect, init_line_status)
    return action_path, grid2op_action_path


def return_action_path(path, actions, debug=False):
    if debug:
        names = [str(action) for action in actions]
    else:
        names = [action.name for action in actions]
    path.remove('init')
    path.remove('end')
    action_path = []
    for node in path:
        if debug:
            id_ = node.split('_t')[0]
        else:
            id_ = int(node.split('_t')[0])
        pos = names.index(id_)
        action_path.append(actions[pos])
    return action_path

def get_grid2op_action_path(action_path, init_topo_vect, init_line_status):
    """
    Compute the grid2op actions equivalent to a given list of OracleAction
    :param action_path: list of OracleAction that define path
    :param init_topo_vect:
    :param init_line_status:
    :return: list of dict representing all grid2op actions that need to be done for optimal path
    """
    grid2op_action_path = []

    # At timestep 0
    grid2op_action = action_path[0].grid2op_action_dict
    grid2op_action_path.append(grid2op_action)

    # At other timesteps: get grid2op transition actions
    for i in range(len(action_path)-1):
        action = action_path[i]
        next_action = action_path[i+1]
        grid2op_action = action.transition_action_to(next_action, init_topo_vect, init_line_status)
        grid2op_action_path.append(grid2op_action)
    return grid2op_action_path
