import warnings

import numpy as np
import pandas as pd

from Episode import EpisodeData
from grid2op.dtypes import dt_float, dt_bool
from oracle4grid.core.agent.OneChangeThenOnlyReconnect import OneChangeThenOnlyReconnect
from oracle4grid.core.graph.attack_graph_module import get_windows_from_df
from oracle4grid.core.reward_computation.Run import Run
from oracle4grid.core.reward_computation.run_many import make_df_from_res


def multiverse_simulation(env, actions, reward_df, debug, env_seed=None, agent_seed=None):
    runs = compute_all_multiverses(env, actions, reward_df, env_seed, agent_seed)
    if len(runs) == 0:
        return reward_df
    multiverse_df = make_df_from_res(runs, debug)
    return pd.concat([reward_df, multiverse_df], ignore_index=True, sort=False)


def compute_all_multiverses(env, actions, reward_df, env_seed=None, agent_seed=None):
    runs = []
    windows = get_windows_from_df(reward_df)
    for window in windows:
        begin = int(window.split("_")[0])
        end = int(window.split("_")[1])
        for attack in windows[window]:
            attack_topos = windows[window][attack]['topos']
            attack = windows[window][attack]['attack']
            attack_topo = attack_topos[0]
            # get the list of topologies that where not computed [all_topo] - windows[u,v][A]
            universes = actions.copy()
            # Remove indexes from already computed topologies, need to sort and reverse the indices to not messup the positions
            for index in sorted(attack_topos, reverse=True):
                del universes[index]
            for universe in universes:
                run = compute_one_multiverse(env, attack_topo, universe, attack, begin, end, env_seed, agent_seed)
                run.action = universe
                run.attacks = [attack for i in range(begin,end+1)]
                run.max_ts = end-begin+1
                run.reset_attacks_id()
                runs.append(run)
    print("Number of multiverse computed :" + str(len(runs)))
    return runs


def compute_one_multiverse(env, topo_init, universe, attack, begin, end, env_seed=None, agent_seed=None):
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        # TODO what do i do if agent cannot do opponent action ?
        combinated_action = universe.grid2op_action + attack
    agent = OneChangeThenOnlyReconnect.gen_next(combinated_action)(env.action_space)
    # set the seed
    if env_seed is not None:
        env.seed(env_seed)
    obs = env.reset()
    if agent_seed is not None:
        agent.seed(agent_seed)
    agent.reset(obs)
    # compute the reward when fast forwarding to t = u-1, doing an action that puts us in T at t = u and stopping at v
    env.fast_forward_chronics(begin)
    episode = init_episode_data(env, end - begin + 1)
    reward = float(env.reward_range[0])
    done = False
    iteration = 0
    time_step = begin
    cum_reward = dt_float(0.0)
    while time_step <= end:
        act = agent.act(obs, reward, done)
        obs, reward, done, info = env.step(act)
        iteration += 1
        time_step += 1
        cum_reward += reward
        opp_attack = env._oppSpace.last_attack
        episode.incr_store(True, iteration, 0,
                           float(reward), env._env_modification,
                           act, obs, opp_attack,
                           info)

    episode.set_meta(env, iteration, float(cum_reward), env_seed, agent_seed)
    res = [(None, None, cum_reward, end - begin + 1, None, episode)]
    return Run(universe, res)


def init_episode_data(env, nb_timestep_max):
    disc_lines_templ = np.full(
        (1, env.backend.n_line), fill_value=False, dtype=dt_bool)

    attack_templ = np.full(
        (1, env._oppSpace.action_space.size()), fill_value=0., dtype=dt_float)
    times = np.full(nb_timestep_max, fill_value=np.NaN, dtype=dt_float)
    rewards = np.full(nb_timestep_max, fill_value=np.NaN, dtype=dt_float)
    actions = np.full((nb_timestep_max, env.action_space.n),
                      fill_value=np.NaN, dtype=dt_float)
    env_actions = np.full(
        (nb_timestep_max, env._helper_action_env.n), fill_value=np.NaN, dtype=dt_float)
    observations = np.full(
        (nb_timestep_max + 1, env.observation_space.n), fill_value=np.NaN, dtype=dt_float)
    disc_lines = np.full(
        (nb_timestep_max, env.backend.n_line), fill_value=np.NaN, dtype=dt_bool)
    attack = np.full((nb_timestep_max, env._opponent_action_space.n), fill_value=0., dtype=dt_float)
    episode = EpisodeData(actions=actions,
                          env_actions=env_actions,
                          observations=observations,
                          rewards=rewards,
                          disc_lines=disc_lines,
                          times=times,
                          observation_space=env.observation_space,
                          action_space=env.action_space,
                          helper_action_env=env._helper_action_env,
                          path_save=None,
                          disc_lines_templ=disc_lines_templ,
                          attack_templ=attack_templ,
                          attack=attack,
                          attack_space=env._opponent_action_space,
                          logger=None,
                          name=env.chronics_handler.get_name(),
                          force_detail=True,
                          other_rewards=[])
    episode.set_parameters(env)
    return episode
