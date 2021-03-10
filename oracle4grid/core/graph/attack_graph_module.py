from itertools import compress
import time
import numpy as np


def filter_attacked_nodes(edges_or, edges_ex, edges_weights, reward_df):
    # attacks_per_topo_df = reward_df[['name', 'attack_id']].drop_duplicates()
    # Transform DF to dict
    # attacks_per_topo = attacks_per_topo_df.set_index('name').T.to_dict('list')
    attacks_per_topo = reward_df.dropna()[['timestep', 'name', 'attack_id']].to_dict("list")
    mask = [is_masked(edge_or, edge_ex, attacks_per_topo) for (edge_or, edge_ex) in zip(edges_or, edges_ex)]
    print("Compressing attack lists...")
    start_time = time.time()
    new_or = list(compress(edges_or, mask))
    new_ex = list(compress(edges_ex, mask))
    new_weights = list(compress(edges_weights, mask))
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    print("number of edges removes because of inconsistency in attacks : " + str(len(edges_or) - len(new_or)))
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
    edge_info = edge.split("_t")
    topo = edge_info[0]
    timestep = int(edge_info[1])
    t_indexes = [i for i, x in enumerate(attacks_per_topo['timestep']) if x == timestep]
    name_indexes = [i for i, x in enumerate(attacks_per_topo['name']) if x == int(topo)]
    final_indexes = set(t_indexes).intersection(name_indexes)
    if len(final_indexes) == 1:
        return attacks_per_topo['attack_id'][final_indexes.pop()]
    elif len(final_indexes) == 0:
        return None
    else:
        raise IndexError("too many values to unpack : " + len(final_indexes))


def get_windows_from_df(reward_df):
    attacks_per_topo = reward_df.dropna()[['timestep', 'name', 'attack_id']].groupby('name')
    all_windows = {}
    for name, name_group in attacks_per_topo:
        first_time = name_group.head(1)['timestep'].iloc[0]
        prev_timestep = int(name_group.head(1)['timestep'].iloc[0]) - 1
        prev_attack = name_group.head(1)['attack_id'].iloc[0]
        for index, row in name_group.iterrows():
            timestep = int(row['timestep'])
            attack_id = row['attack_id']
            if (timestep - prev_timestep != 1) or prev_attack != attack_id:
                # Save block
                all_windows = _save_in_window(all_windows, first_time, prev_timestep, prev_attack, name)
                # Reset block
                prev_attack = attack_id
                first_time = timestep
            prev_timestep = timestep
        # Save block
        all_windows = _save_in_window(all_windows, first_time, prev_timestep, prev_attack, name)
    return all_windows


def _save_in_window(all_windows, first_time, prev_timestep, prev_attack, name):
    key = str(first_time) + '_' + str(prev_timestep)
    if key not in all_windows:
        all_windows[key] = {}
    if prev_attack not in all_windows[key]:
        all_windows[key][prev_attack] = []
    all_windows[key][prev_attack].append(name)
    return all_windows
