
def generate(atomic_actions) :
    ret = []
    all_actions = generate_all(atomic_actions)
    for action in all_actions :
        if keep(action):
            ret.append(action)
    return ret

def generate_all(atomic_actions) :


def run_pf_check(action) :


def keep(action) :
    return run_pf_check()
