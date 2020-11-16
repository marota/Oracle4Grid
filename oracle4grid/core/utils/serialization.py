import os
import pickle
import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def serialize(obj, name, dir):
    outfile = open(os.path.join(dir, name + ".pkl"), 'wb')
    pickle.dump(obj, outfile)
    outfile.close()


def load_serialized_object(name, dir):
    infile = open(os.path.join(dir, name + ".pkl"), 'rb')
    obj = pickle.load(infile)
    infile.close()
    return obj


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
            prefix = node.split('_t')[0]
            timestep = int(node.split('_t')[1])
            x = x_axis[timestep + 1]
            y = y_axis[prefix]
        layout[node] = np.array([x, y])

    ## Plot graph with its layout
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
