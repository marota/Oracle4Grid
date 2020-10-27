import networkx as nx

def generate(reward_df, config, env, actions):
    # Compute possible transitions list for each action
    reachable_topologies = get_reachable_topologies(actions, env.action_space)

    # Build graph
    graph = build_transition_graph(reachable_topologies, reward_df)
    return graph

def get_reachable_topologies(actions, action_space):
    action_couples = [(action1, action2) for action1 in actions for action2 in actions]
    modified_subs = [action_couple[0].transition_bus_action_to(action_couple[1])[1]
                     for action_couple in action_couples]
    # Filter tout ce qui est > limite
    # Transformer au format [[indices reachables par i]...]

    # Idem pour les lines
    return 0

def build_transition_graph(reachable_topologies, reward_df):
    return 0
