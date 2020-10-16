# util to generate a atomic action or a list of atomic actions

def get_atomic_actions_names(atomic_actions) :
    # Name of each target configuration
    named_atomic_actions = {}
    for key in atomic_actions:
        if 'sub' in key:
            dict_sub = {f'{key}_{c}': [key, atomic_actions[key][c]] for c in atomic_actions[key]}
            named_atomic_actions.update(dict_sub)
    return  named_atomic_actions
