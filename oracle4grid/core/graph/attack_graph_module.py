from itertools import compress
import time
import numpy as np
import itertools
import networkx as nx


def get_info_from_edge(edge):
    split_1 = edge.split("_t")
    topo = split_1[0]
    split_2 = split_1[1].split("_atk")[0]
    timestep = split_2[0]
    attack = None
    if len(split_2) == 2:
        attack = split_2[1]
    return topo, timestep, attack


def get_windows_from_df(reward_df):
    #Pre computing the correspondig attack to attack_id
    attack_df = reward_df[['attack_id', 'attacks']].drop_duplicates(subset='attack_id')
    attack_dict = attack_df.set_index('attack_id').to_dict()['attacks']

    all_windows = {}

    pivot = reward_df[['timestep', 'attack_id', 'name', 'node_name']]\
        .pivot(index='timestep', columns=['name', 'node_name'], values='attack_id')
    serie = pivot.sum(axis=1)
    non_zero = serie[serie != 0]
    if non_zero.first_valid_index() is None:
        #No attacks in runs
        return all_windows
    prevIndex = np.nan
    begin = None
    end = None
    for index, value in non_zero.items():
        if index != prevIndex + 1:
            if begin is not None:
                window = str(begin)+'_'+str(prevIndex)
                all_windows[window] = _save_all_in_window(begin, prevIndex, pivot, attack_dict, reward_df)
            begin = index
        prevIndex = index
    window = str(begin)+'_'+str(prevIndex)
    all_windows[window] = _save_all_in_window(begin, prevIndex, pivot, attack_dict,reward_df)
    return all_windows


def _save_all_in_window(begin, end, pivot, attack_dict, reward_df):
    # get all attacks per topo in the given range, by taking the first valid value in the range
    # attacks_per_topo = pivot[pivot.index.isin(range(begin, end+1))].dropna(axis=1).apply(lambda col: col.loc[col.first_valid_index()])
    a_p_t = reward_df[reward_df['timestep']
        .isin(range(begin, end+1))][['name', 'attack_id']]\
        .drop_duplicates().to_dict(orient='records')
    window_serial = {}
    for item in a_p_t:
        topo = item['name']
        attack = item['attack_id']
    #for topo, attack in attacks_per_topo.items():
        if attack not in window_serial:
            window_serial[attack] = {}
            window_serial[attack]['topos'] = []
            window_serial[attack]['attack'] = attack_dict[attack]
        window_serial[attack]['topos'].append(topo)
    return window_serial
