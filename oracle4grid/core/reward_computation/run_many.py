# outputs a dataframe with rewards for each run (run = N actions) and each timestep
# takes a dict of all combinations of actions
from multiprocessing import Pool

import pandas
from tqdm import tqdm

from oracle4grid.core.reward_computation.run_one import run_one


def run_all(actions, env, max_iter=1, nb_process=1, debug=False,agent_seed=None,env_seed=None):
    if debug:
        print('\n')
        print("============== 2 - Rewards simulation ==============")
    if nb_process is 1:
        all_res = serie(env, actions, max_iter,agent_seed,env_seed)
    else:
        all_res = parallel(env, actions, max_iter, nb_process,agent_seed,env_seed)
    return make_df_from_res(all_res)


def make_df_from_res(all_res):
    cols = ["action", "timestep", "reward", "overload_reward"]
    data = []

    for run in all_res:
        # Temporary fix in case there is divergence - rewards and other_rewards should be under the same convention (length, NaN)
        run = check_other_rewards(run)
        for t in range(run.rewards.shape[0]):
            data.append({"action": run.action, "timestep": t, "reward": run.rewards[t], "overload_reward": run.other_rewards[t]["overload_reward"]})
    df = pandas.DataFrame(data, columns=cols)
    return df

def check_other_rewards(run):
    other_rewards = run.other_rewards.copy()
    ref_dict = other_rewards[0]
    n_other_rewards = len(other_rewards)
    if n_other_rewards < run.max_ts:
        ref_dict_nan = {key: float('nan') for key in ref_dict.keys()}
        other_rewards += [ref_dict_nan for i in range(run.max_ts-n_other_rewards)]
        run.other_rewards = other_rewards
    return run

def serie(env, actions, max_iter,agent_seed=None,env_seed=None):
    all_res = []
    with tqdm(total=len(actions)) as pbar:
        for action in actions:
            all_res.append(run_one(action, env, max_iter,agent_seed,env_seed))
            pbar.update(1)
    return all_res


def parallel(env, actions, max_iter,nb_process,agent_seed=None,env_seed=None):
    all_res = []
    with tqdm(total=len(actions)) as pbar:
        with Pool(nb_process) as p:
            runs = p.starmap(run_one,
                             [(action, env, max_iter,agent_seed,env_seed) for action in actions])
            for run in runs:
                all_res.append(run)
            pbar.update(1)
    return all_res
