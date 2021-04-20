import json
import pandas as pd
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from oracle4grid.core.graph.attack_graph_module import get_info_from_edge


def draw_graph(graph, max_iter, save=None):
    layout = {}

    # Each unique prefix (Action) is given a y
    prefixes = {node.split('_t')[0] for node in graph.nodes}
    y_axis = np.linspace(start=-1, stop=1, num=len(prefixes))
    y_axis = {prefix: y for prefix, y in zip(prefixes, y_axis)}

    # Each timestep is given a x
    x_axis = np.linspace(start=-1, stop=1, num=(max_iter + 2))

    # Each node of the graph is given a x and y
    for node in graph.nodes:
        if node == 'init':
            x = -1
            y = 0
        elif node == 'end':
            x = 1
            y = 0
        else:
            prefix, timestep, attack = get_info_from_edge(node)
            x = x_axis[timestep + 1]
            y = y_axis[prefix]
        layout[node] = np.array([x, y])

    # Plot graph with its layout
    fig, ax = plt.subplots(1, 1, figsize=(25, 15))
    # Graph structure
    nx.draw_networkx(graph, pos=layout, ax=ax)
    # Rounded labels
    labels = nx.get_edge_attributes(graph, 'weight')
    for k, v in labels.items():
        labels[k] = round(v, 2)
    nx.draw_networkx_edge_labels(graph, pos=layout, edge_labels=labels, font_size=9, alpha=0.6)
    if save is not None:
        fig.savefig(os.path.join(save, "graphe.png"))


def display_topo_count(best_path, dir, n_best=10, name = None):
    best_path_df = pd.Series(best_path)
    topo_count = best_path_df.value_counts().head(n_best)
    fig, ax = plt.subplots(1, 1, figsize=(22, 15))
    topo_count.plot.bar(ax=ax)
    if name is None:
        name = 'best_path_topologies_count.png'
    fig.savefig(os.path.join(dir, name))
    return topo_count


def serialize_reward_df(reward_df, dir):
    df = reward_df.copy()
    df = df.set_index(['action', 'timestep'])
    df = df.unstack(level=0)['reward']
    df.to_csv(os.path.join(dir, "reward_df.csv"), sep=';', index=True)

def serialize_graph(graph, dir):
    edge_list = nx.to_pandas_edgelist(graph)
    edge_list.to_csv(os.path.join(dir, "edge_list.csv"), sep=';', index=False)

def serialize(obj, name, dir, format='pickle'):
    if format == 'pickle':
        outfile = open(os.path.join(dir, name + ".pkl"), 'wb')
        pickle.dump(obj, outfile)
        outfile.close()
    elif format == 'json':
        outfile = open(os.path.join(dir, name + ".json"), 'w')
        json.dump(obj, outfile)
        outfile.close()


def load_serialized_object(name, dir):
    infile = open(os.path.join(dir, name + ".pkl"), 'rb')
    obj = pickle.load(infile)
    infile.close()
    return obj
