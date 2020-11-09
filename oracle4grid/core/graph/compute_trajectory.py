# Use algo to find best path
import networkx as nx


SHORTEST = "shortest"
LONGEST = "longest"


def best_path(graph, best_path_type, actions, debug = False):
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
    action_path = return_action_path(path, actions)
    return action_path

def return_action_path(path, actions):
    names = [action.name for action in actions]
    path.remove('init')
    path.remove('end')
    action_path = []
    for node in path:
        id_ = int(node.split('_t')[0])
        pos = names.index(id_)
        action_path.append(actions[pos])
    return action_path
