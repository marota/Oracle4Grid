from itertools import compress


def filter_attacked_nodes(edges_or, edges_ex, edges_weights, reward_df):
    mask = []
    new_or, new_ex, new_weights = [], [], []
    for i, edge_or in enumerate(edges_or):
        edge_ex = edges_ex[i]
        mask.append(is_masked(edge_or, edge_ex, reward_df))
    new_or = list(compress(edges_or, mask))
    new_ex = list(compress(edges_ex, mask))
    new_weights = list(compress(edges_weights, mask))
    return new_or, new_ex, new_weights


def is_masked(edge_or, edge_ex, reward_df):
    if edge_or is "init" or edge_ex is "end" or edge_or is "end" or edge_ex is "init":
        return True
    # OR data
    or_attack = get_attack_from_edge(edge_or, reward_df)
    # EX data
    ex_attack = get_attack_from_edge(edge_ex, reward_df)
    return or_attack == ex_attack


def get_attack_from_edge(edge, reward_df):
    topo = edge.split("_t")[0]
    timestep = int(edge.split("_t")[1])
    df_line = reward_df[(reward_df["timestep"] == timestep) & (reward_df["name"] == topo)].iloc[0]
    attack = df_line["attack_id"]
    return attack