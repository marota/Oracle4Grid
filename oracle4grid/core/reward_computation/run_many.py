# outputs a dataframe with rewards for each run (run = N actions) and each timestep
# takes a dict of all combinations of actions
from multiprocessing import Pool

import pandas
from pandas import notnull
from tqdm import tqdm

from oracle4grid.core.reward_computation.run_one import run_one


def run_all(actions, env, max_iter=1, nb_process=1, debug=False, agent_seed=None, env_seed=None):
    if debug:
        print('\n')
        print("============== 2 - Rewards simulation ==============")
    if agent_seed is not None:  # Grid2op runner expect a list of seeds
        agent_seed=[agent_seed]
    if env_seed is not None:
        env_seed=[env_seed]
    if nb_process is 1:
        all_res = serie(env, actions, max_iter, agent_seed, env_seed)
    else:
        all_res = parallel(env, actions, max_iter, nb_process, agent_seed, env_seed)
    return make_df_from_res(all_res, debug)


def make_df_from_res(all_res, debug, multiverse=False):
    cols = ["action", "timestep", "reward", "overload_reward"
            , "attacks", "attack_id", "name", "node_name", "multiverse"
            ]
    data = []
    for run in all_res:
        # Temporary fix in case there is divergence - rewards and other_rewards should be under the same convention (length, NaN)
        run = check_other_rewards(run)
        for n, t in enumerate(range(run.begin_ts, run.begin_ts+run.nb_timestep)):
            data.append(to_json(run, t, n,  debug, multiverse))
    df = pandas.DataFrame(data, columns=cols)
    return df


def check_other_rewards(run):
    other_rewards = run.other_rewards.copy()
    n_other_rewards = len(other_rewards)
    #If divergence on first timestep, we need to create a fake other reward
    if n_other_rewards == 0:
        run.other_rewards = [{'overload_reward': float('nan')}]
    else:
        ref_dict = other_rewards[0]
        if n_other_rewards < run.nb_timestep:
            ref_dict_nan = {key: float('nan') for key in ref_dict.keys()}
            other_rewards += [ref_dict_nan for i in range(run.max_ts - n_other_rewards)]
            run.other_rewards = other_rewards
    return run


def to_json(run, t, n, debug, multiverse):
    name = get_action_name(run.action, debug)
    return {
        "action": run.action,
        "timestep": t,
        "reward": run.rewards[n],
        "overload_reward": run.other_rewards[n]["overload_reward"],
        "attacks": run.attacks[n],
        "attack_id": run.attacks_id[n],
        "name": name,
        "node_name": get_node_name(name, t, run.attacks_id[n]),
        "multiverse": multiverse
    }


def get_action_name(action, debug):
    return str(action) if debug else action.name


def get_node_name(name, t, attack_id):
    attack_label = ("_atk" + str(attack_id)) if notnull(attack_id) else ""
    return str(name) + "_t" + str(t) + attack_label


def serie(env, actions, max_iter, agent_seed=None, env_seed=None):
    all_res = []
    with tqdm(total=len(actions)) as pbar:
        for action in actions:
            all_res.append(run_one(action, env.get_params_for_runner(), max_iter, agent_seed, env_seed))
            pbar.update(1)
    return all_res


def parallel(env, actions, max_iter, nb_process, agent_seed=None, env_seed=None):
    all_res = []
    with tqdm(total=len(actions)) as pbar:
        with Pool(nb_process) as p:
            runs = p.starmap(run_one,
                             [(action, env.get_params_for_runner(), max_iter, agent_seed, env_seed) for action in actions])
            for run in runs:
                all_res.append(run)
            pbar.update(1)
    return all_res
