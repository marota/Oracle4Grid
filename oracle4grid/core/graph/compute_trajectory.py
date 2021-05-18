# Use algo to find best path
import networkx as nx
import numpy as np

from oracle4grid.core.graph.attack_graph_module import get_info_from_edge


SHORTEST = "shortest"
LONGEST = "longest"


def best_path(graph, best_path_type, actions, debug = False):
    if debug:
        print('\n')
        print("============== 4 - Computation of best action path ==============")
    path = None
    if not nx.has_path(graph, 'init', 'end'):

        l_nodes_with_init = list(graph.subgraph(nx.shortest_path(graph.to_undirected(), 'init')).nodes)#get connected component to init node
        l_nodes_with_init.remove('init')
        if len(l_nodes_with_init) == 0:
            max_timestep = 0
        else:
            timesteps=[int(get_info_from_edge(s)[1]) for s in l_nodes_with_init]#get timesteps of connected component
            max_timestep=np.max(timesteps)
        print("WARNING: there is no path without overloads between t0 and max iter. Problem occurs at timestep "+ str(max_timestep))

        return [], [], []
    else:
        if best_path_type == SHORTEST:
            path = nx.bellman_ford_path(graph, 'init', 'end')  # shortest_path(G)
        elif best_path_type == LONGEST:
            path = nx.dag_longest_path(graph)  # Longest path
        # Add more treatment here

        # Traduce path in terms of OracleActions
        action_path = return_action_path(path, actions, debug=debug)
        grid2op_action_path = get_grid2op_action_path(action_path)
        return path, action_path, grid2op_action_path


def best_path_no_overload(graph, best_path_type, actions, debug = False):
    if debug:
        print('\n')
        print("============== 4 - Computation of best action path with no overload ==============")

    path = None
    graph_no_overload = graph.copy()
    nodes = list(graph_no_overload.nodes())
    for node in nodes:
        if node is not "init" and node is not "end" and "overload_reward" in graph.nodes[node] and graph.nodes[node]["overload_reward"] == 0:
            graph_no_overload.remove_node(node)

    return best_path(graph_no_overload, best_path_type, actions, debug)


def return_action_path(path, actions, debug=False):
    if debug:
        names = [str(action) for action in actions]
    else:
        names = [action.name for action in actions]
    if 'init' in path:
        path.remove('init')
    if 'end' in path:
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

def get_grid2op_action_path(action_path):
    """
    Compute the grid2op actions equivalent to a given list of OracleAction
    :param action_path: list of OracleAction that define path
    :return: list of dict representing all grid2op actions that need to be done for optimal path
    """
    grid2op_action_path = []

    # At timestep 0
    grid2op_action = action_path[0].grid2op_action
    grid2op_action_path.append(grid2op_action)

    # At other timesteps: get grid2op transition actions
    for i in range(len(action_path)-1):
        action = action_path[i]
        next_action = action_path[i+1]
        grid2op_action = action.transition_action_to(next_action)
        grid2op_action_path.append(grid2op_action)
    return grid2op_action_path
