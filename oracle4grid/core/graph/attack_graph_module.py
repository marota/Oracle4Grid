from itertools import compress
import time


def filter_attacked_nodes(edges_or, edges_ex, edges_weights, reward_df):
    attacks_per_topo_df = reward_df[['name', 'attack_id']].drop_duplicates()
    # Transform DF to dict
    attacks_per_topo = attacks_per_topo_df.set_index('name').T.to_dict('list')
    mask = [is_masked(edge_or, edge_ex, attacks_per_topo) for (edge_or, edge_ex) in zip(edges_or, edges_ex)]
    print("Compressing attack lists...")
    start_time = time.time()
    new_or = list(compress(edges_or, mask))
    new_ex = list(compress(edges_ex, mask))
    new_weights = list(compress(edges_weights, mask))
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    return new_or, new_ex, new_weights


def is_masked(edge_or, edge_ex, attacks_per_topo):
    if edge_or is "init" or edge_ex is "end" or edge_or is "end" or edge_ex is "init":
        return True
    # OR data
    or_attack = get_attack_from_edge(edge_or, attacks_per_topo)
    # EX data
    ex_attack = get_attack_from_edge(edge_ex, attacks_per_topo)
    return or_attack is None or ex_attack is None or or_attack == ex_attack


def get_attack_from_edge(edge, attacks_per_topo):
    topo = edge.split("_t")[0]
    return attacks_per_topo[topo]
