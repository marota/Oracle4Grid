from itertools import compress
import time
import numpy as np
import itertools
import networkx as nx


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

def filter_attack_edges(reachable_topologies,topo_ordered_names,reward_df,max_iter,n_init,n_end):
    """
    Other possible function to filter edges that don't share the attack id
    """
    print("Removing attack edges")
    # To filter graph edges whose nodes don't share the same attack (if attack exist)
    attacks_table = reward_df[['timestep', 'attack_id', 'name']].pivot(index='timestep', columns='name',
                                                                       values='attack_id')
    attacks_table = attacks_table.reset_index(drop=True).rename_axis(None,
                                                                     axis=1)  # to have a proper data table without parasite names

    edges_attack_diff = list(
        itertools.chain(*[attacks_table[reachable_topologies[i]][1:max_iter].reset_index(drop=True).
                        sub(attacks_table[topo_ordered_names[i]][0:max_iter - 1], axis=0).values.flatten(order='F')
                          for i in range(len(topo_ordered_names))] ))



    edges_attack_filtered_after_init_befor_end = [False if (np.isnan(edges_attack_diff[i]) or edges_attack_diff[i] == 0) else True for i in
                              range(len(edges_attack_diff))]
    #add filter for init and end node
    edges_attack_filtered_init = [False for i in range(n_init)]
    edges_attack_filtered_end=[False for i in range(n_end)]

    #number of edges normally filtered on the test case,by looking at reachable topology for each topology and attacks per topology at timestep 400
    # 1*48+1*48+1*48+1*48+1*48+1*48+1*48+1*48=8*48=384 over test case, for 2nd window
    edges_attack_filtered = edges_attack_filtered_init + edges_attack_filtered_after_init_befor_end + edges_attack_filtered_end
    print("number of edges removes because of inconsistency in attacks : " + str(np.sum(edges_attack_filtered)))
    return edges_attack_filtered


def is_masked(edge_or, edge_ex, attacks_per_topo):
    if edge_or is "init" or edge_ex is "end" or edge_or is "end" or edge_ex is "init":
        return True
    # OR data
    or_attack = get_attack_from_edge(edge_or, attacks_per_topo)
    # EX data
    ex_attack = get_attack_from_edge(edge_ex, attacks_per_topo)
    return or_attack is None or ex_attack is None or or_attack == ex_attack


def get_info_from_edge(edge):
    split_1 = edge.split("_t")
    topo = split_1[0]
    split_2 = split_1[1].split("_atk")[0]
    timestep = split_2[0]
    attack = None
    if len(split_2) == 2:
        attack = split_2[1]
    return topo, timestep, attack

def get_attack_from_edge(edge, attacks_per_topo):
    topo, timestep, attack = get_info_from_edge(edge)
    t_indexes = [i for i, x in enumerate(attacks_per_topo['timestep']) if x == timestep]
    #name_indexes = [i for i, x in enumerate(attacks_per_topo['name']) if x == int(topo)]#x=topo
    name_indexes = [i for i, x in enumerate(attacks_per_topo['name']) if str(x) == str(topo)]
    final_indexes = set(t_indexes).intersection(name_indexes)
    if len(final_indexes) == 1:
        return attacks_per_topo['attack_id'][final_indexes.pop()]
    elif len(final_indexes) == 0:
        return None
    else:
        raise IndexError("too many values to unpack : " + len(final_indexes))


def get_windows_from_df(reward_df):
    """

    :param reward_df: The reward dataframe given by computation module
    :return: A dict containing all windows "begin, end" associated to all attacks , and all topologies per attacks
    """
    attacks_per_topo = reward_df.dropna()[['timestep', 'name', 'attacks', 'attack_id']].groupby('name')
    all_windows = {}
    for name, name_group in attacks_per_topo:
        first_time = name_group.head(1)['timestep'].iloc[0]
        prev_timestep = int(name_group.head(1)['timestep'].iloc[0]) - 1
        prev_attack_id = name_group.head(1)['attack_id'].iloc[0]
        prev_attack = name_group.head(1)['attacks'].iloc[0]
        for index, row in name_group.iterrows():
            timestep = int(row['timestep'])
            attack_id = row['attack_id']
            if (timestep - prev_timestep != 1) or prev_attack_id != attack_id:
                # Save block
                all_windows = _save_in_window(all_windows, first_time, prev_timestep, prev_attack_id, prev_attack, name)
                # Reset block
                prev_attack_id = attack_id
                prev_attack = row['attacks']
                first_time = timestep
            prev_timestep = timestep
        # Save block
        all_windows = _save_in_window(all_windows, first_time, prev_timestep, prev_attack_id, prev_attack, name)
    return all_windows


def _save_in_window(all_windows, first_time, prev_timestep, prev_attack_id, prev_attack, name):
    key = str(first_time) + '_' + str(prev_timestep)
    if key not in all_windows:
        all_windows[key] = {}
    if prev_attack_id not in all_windows[key]:
        all_windows[key][prev_attack_id] = {}
        all_windows[key][prev_attack_id]['topos'] = []
        all_windows[key][prev_attack_id]['attack'] = prev_attack
    all_windows[key][prev_attack_id]['topos'].append(name)
    return all_windows
