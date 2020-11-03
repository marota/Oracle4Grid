# Use algo to find best path
import networkx as nx


SHORTEST = "shortest"
LONGEST = "longest"


def best_path(graph, best_path_type):
    path = None
    if best_path_type == SHORTEST:
        path = nx.bellman_ford_path(graph, 'init', 'end')  # shortest_path(G)
    elif best_path_type == LONGEST:
        path = nx.dag_longest_path(graph)  # Longest path
    # Add more treatment here
    return path
