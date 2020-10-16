#should output a dict with all combinations of actions based on dict of possible actions
from tqdm import tqdm

def generate(atomic_actions, depth) :
    ret = []
    all_actions = generate_all(atomic_actions, depth)
    for action in all_actions :
        if keep(action):
            ret.append(action)
    return ret

def generate_all(atomic_actions) :
    # don't generate associative actions

def run_pf_check(action) :



def keep(action) :
    return run_pf_check(action)
