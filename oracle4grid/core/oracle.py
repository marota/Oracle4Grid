import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from oracle4grid.core.graph import graph_generator, compute_trajectory, indicators
from oracle4grid.core.utils.prepare_environment import get_initial_configuration
from oracle4grid.core.reward_computation import run_many
from oracle4grid.core.actions_utils import combinator
from oracle4grid.core.utils.config_ini_utils import MAX_ITER, MAX_DEPTH, NB_PROCESS, N_TOPOS
from oracle4grid.core.utils.serialization import draw_graph


def oracle(atomic_actions, env, debug, config, debug_directory=None):
    # 0 - Preparation : Get initial topo and line status
    init_topo_vect, init_line_status = get_initial_configuration(env)

    # 1 - Action generation step
    actions = combinator.generate(atomic_actions, int(config[MAX_DEPTH]), env, debug, init_topo_vect, init_line_status)

    # 2 - Actions rewards simulation
    reward_df = run_many.run_all(actions, env, int(config[MAX_ITER]), int(config[NB_PROCESS]), debug=debug)
    if debug:
        print(reward_df)
        serialize_reward_df(reward_df, debug_directory)
        # serialize(reward_df, name='reward_df', dir=debug_directory)
        # reward_df = load_serialized_object('reward_df', debug_directory)

    # 3 - Graph generation
    graph = graph_generator.generate(reward_df, init_topo_vect, init_line_status, int(config[MAX_ITER]), debug=debug)
    if debug:
        # serialize(graph, name="graphe", dir=debug_directory)
        draw_graph(graph, int(config[MAX_ITER]), save=debug_directory)

    # 4 - Best path computation (returns actions.npz + a list of atomic action dicts??)
    best_path, grid2op_action_path = compute_trajectory.best_path(graph, config["best_path_type"], actions,
                                             init_topo_vect, init_line_status, debug = debug)

    if debug:
        print(best_path)
        # Serialization for agent replay
        serialize(grid2op_action_path, 'best_path_grid2op_action',
                  dir=debug_directory, format='pickle')
        topo_count = display_topo_count(best_path, dir = debug_directory)
        print('10 best topologies in optimal path')
        print(topo_count)


    # 5 - Indicators computation
    kpis = indicators.generate(best_path, reward_df, config["best_path_type"], int(config[N_TOPOS]), debug=debug)
    if debug:
        print(kpis)
        kpis.to_csv(os.path.join(debug_directory, "kpis.csv"), sep=';', index=False)

    return best_path, grid2op_action_path

def display_topo_count(best_path, dir, n_best = 10):
    best_path_df = pd.Series(best_path)
    topo_count = best_path_df.value_counts().head(n_best)
    fig, ax = plt.subplots(1, 1, figsize=(22, 15))
    topo_count.plot.bar(ax = ax)
    fig.savefig(os.path.join(dir, "best_path_topologies_count.png"))
    return topo_count

def serialize_reward_df(reward_df, dir):
    df = reward_df.copy()
    df = df.set_index(['action','timestep'])
    df = df.unstack(level=0)['reward']
    df.to_csv(os.path.join(dir, "reward_df.csv"), sep=';', index=True)

def serialize(obj, name, dir, format = 'pickle'):
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

def draw_graph(graph, max_iter, save = None):
    layout = {}

    # Each unique prefix (Action) is given a y
    prefixes = {node.split('_t')[0] for node in graph.nodes}
    y_axis = np.linspace(start = -1, stop = 1, num = len(prefixes))
    y_axis = {prefix:y for prefix,y in zip(prefixes,y_axis)}

    # Each timestep is given a x
    x_axis = np.linspace(start = -1, stop = 1, num = (max_iter+2))

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
            x = x_axis[timestep+1]
            y = y_axis[prefix]
        layout[node] = np.array([x,y])

    ## Plot graph with its layout
    fig, ax = plt.subplots(1, 1, figsize=(22, 12))
    # Graph structure
    nx.draw_networkx(graph, pos = layout, ax = ax, font_size = 11, alpha = 0.8)
    # Rounded labels
    labels = nx.get_edge_attributes(graph, 'weight')
    for k, v in labels.items():
        labels[k] = round(v, 2)
    nx.draw_networkx_edge_labels(graph, pos = layout, edge_labels=labels, font_size = 7, alpha = 0.6)
    if save is not None:
       fig.savefig(os.path.join(save,"graphe.png"))

