import pandas as pd
import networkx as nx

from oracle4grid.core.utils.constants import DICT_GAME_PARAMETERS


def generate(reward_df, config, init_topo_vect, init_line_status):

    # Compute possible transitions list for each action
    reachable_topologies = get_reachable_topologies(reward_df, init_topo_vect, init_line_status)

    # Build graph
    graph = build_transition_graph(reachable_topologies, reward_df, config)
    return graph


def get_reachable_topologies(reward_df, init_topo_vect, init_line_status):
    actions = reward_df['action'].unique()
    action_couples = [(action1, action2) for action1 in actions for action2 in actions]
    modified_subs = [len(action_couple[0].modified_subs_to(action_couple[1], init_topo_vect))
                     for action_couple in action_couples]
    modified_lines = [len(action_couple[0].modified_lines_to(action_couple[1], init_line_status))
                     for action_couple in action_couples]

    # Filter tout ce qui est > limite
    valid_action_couples = [action_couple for action_couple, n_subs, n_lines in zip(action_couples, modified_subs, modified_lines)
                            if (n_subs <= DICT_GAME_PARAMETERS["MAX_SUB_CHANGED"] and n_lines<=DICT_GAME_PARAMETERS["MAX_LINE_STATUS_CHANGED"])]

    # Formattage
    reachable_topologies = []
    for action in actions:
        reachable_topologies_from_action = [action_couple[1].name for action_couple in valid_action_couples if action_couple[0].name == action.name]
        reachable_topologies.append(reachable_topologies_from_action)
    return reachable_topologies


def build_transition_graph(reachable_topologies, reward_df, config):
    # We assume in this fonction that all actions are convergent and that reachable_topologies and reward_df are ordered the same way
    duration = int(config['max_depth'])
    reward_df['name'] = [action.name for action in reward_df['action']]
    convergentActions = reward_df['name'].unique()


    ## Compute edge origins and extremities for each timestep x possible transition
    edgeNamesOr = [str(convergentActions[i]) + '_t_' + str(int(t)) for t in range(duration) for i in
                   range(len(reachable_topologies))
                   for j in reachable_topologies[i]]
    edgeNamesEx = [str(j) + '_t_' + str(int(t + 1)) for t in range(duration) for i in
                   range(len(reachable_topologies))
                   for j in reachable_topologies[i]]
    EdgeWeight = [reward_df.loc[(reward_df['timestep']==t+1)&(reward_df['name']==j), 'reward'].values[0] for t in range(duration) for i in
                  range(len(reachable_topologies))
                  for j in reachable_topologies[i]]

    ## Add symbolic init and end node
    edgeNamesOrNodeSource = ['init' for j in reachable_topologies]
    edgeNamesExNodeSource = [str(convergentActions[i]) + '_t_' + str(0) for i in range(len(reachable_topologies))]
    EdgeWeightNodeSource = [reward_df.loc[(reward_df['timestep']==0)&(reward_df['name']==j), 'reward']
                            for j in range(len(reachable_topologies))]
    edgeNamesExNodeEnd = ['end' for j in reachable_topologies]
    edgeNamesOrNodeEnd = [str(convergentActions[i]) + '_t_' + str(int(duration)) for i in
                          range(len(reachable_topologies))]
    EdgeWeightNodeEnd = [0.1 for j in reachable_topologies]


    ## Create graph object
    edgeNamesOr = edgeNamesOrNodeSource + edgeNamesOr + edgeNamesOrNodeEnd
    edgeNamesEx = edgeNamesExNodeSource + edgeNamesEx + edgeNamesExNodeEnd
    EdgeWeight = EdgeWeightNodeSource + EdgeWeight + EdgeWeightNodeEnd
    edgeDf = pd.DataFrame({'or': edgeNamesOr, 'ex': edgeNamesEx, 'weight': EdgeWeight})
    G = nx.from_pandas_edgelist(edgeDf, target='ex', source='or', edge_attr=['weight'], create_using=nx.DiGraph())
    return G
