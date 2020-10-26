# outputs a dataframe with rewards for each run (run = N actions) and each timestep
# takes a dict of all combinations of actions
from multiprocessing import Pool

import pandas
from tqdm import tqdm

from oracle4grid.core.reward_computation.run_one import run_one


def run_all(actions, env, max_iter=1, nb_process=1):
    if nb_process is 1:
        all_res = serie(env, actions, max_iter)
    else:
        all_res = parallel(env, actions, max_iter, nb_process)
    df = make_df_from_res(all_res)
    return df


def make_df_from_res(all_res):
    cols = ["action", "id", "cum_reward", "nb_timesteps", "episode_data"]
    df = pandas.DataFrame(all_res, columns=cols)
    return df


def serie(env, actions, max_iter):
    all_res = []
    with tqdm(total=len(actions)) as pbar:
        for action in actions:
            all_res.append(run_one(action, env, max_iter))
            pbar.update(1)
    return all_res


def parallel(env, actions, max_iter, nb_process):
    all_res = []
    with tqdm(total=len(actions)) as pbar:
        with Pool(nb_process) as p:
            runs = p.starmap(run_one,
                             [(action, env, max_iter) for action in actions])
            for run in runs:
                all_res.append(run)
            pbar.update(1)
    return all_res
