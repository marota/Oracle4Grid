import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from oracle4grid.core.graph import graph_generator, compute_trajectory, indicators
from oracle4grid.core.utils.prepare_environment import get_initial_configuration
from oracle4grid.core.reward_computation import run_many
from oracle4grid.core.actions_utils import combinator
from oracle4grid.core.utils.config_ini_utils import MAX_ITER, MAX_DEPTH, NB_PROCESS


# runs all steps one by one
# handles visualisation in each step

def oracle(atomic_actions, env, debug, config, debug_directory=None):

    # 0 - Preparation : Get initial topo and line status
    init_topo_vect, init_line_status = get_initial_configuration(env)

    # 1 - Action generation step
    actions = combinator.generate(atomic_actions, int(config[MAX_DEPTH]), env, debug, init_topo_vect, init_line_status)

    # 2 - Actions rewards simulation
    reward_df = run_many.run_all(actions, env, int(config[MAX_ITER]), int(config[NB_PROCESS]))
    if debug:
        print(reward_df)
        reward_df.to_csv(os.path.join(debug_directory, "reward_df.csv"), sep = ';', index = False)
        #serialize(reward_df, name='reward_df', dir=debug_directory)
        #reward_df = load_serialized_object('reward_df', debug_directory)

    # 3 - Graph generation
    graph = graph_generator.generate(reward_df, init_topo_vect, init_line_status, int(config[MAX_ITER]), debug = debug)
    if debug:
        # serialize(graph, name="graphe", dir=debug_directory)
        draw_graph(graph, int(config[MAX_ITER]), save = debug_directory)

    # 4 - Best path computation (returns actions.npz + a list of atomic action dicts??)
    best_path = compute_trajectory.best_path(graph, config["best_path_type"], actions, debug = debug)
    if debug:
        print(best_path)

    # 5 - Indicators computation
    kpis = indicators.generate(graph, reward_df, debug = debug)
    if debug:
        print(kpis)

    return best_path


def serialize(obj, name, dir):
    outfile = open(os.path.join(dir, name + ".pkl"), 'wb')
    pickle.dump(obj, outfile)
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
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    # Graph structure
    nx.draw_networkx(graph, pos = layout, ax = ax)
    # Rounded labels
    labels = nx.get_edge_attributes(graph, 'weight')
    for k, v in labels.items():
        labels[k] = round(v, 2)
    nx.draw_networkx_edge_labels(graph, pos = layout, edge_labels=labels, font_size = 9, alpha = 0.6)
    if save is not None:
       fig.savefig(os.path.join(save,"graphe.png"))
