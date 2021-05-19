import warnings

import numpy as np
import pandas as pd

from grid2op.Episode import EpisodeData
from grid2op.dtypes import dt_float, dt_bool
from oracle4grid.core.agent.OneChangeThenOnlyReconnect import OneChangeThenOnlyReconnect
from oracle4grid.core.graph.attack_graph_module import get_windows_from_df
from oracle4grid.core.reward_computation.Run import Run
from oracle4grid.core.reward_computation.run_many import make_df_from_res, get_action_name


def multiverse_simulation(env, actions, reward_df, debug, env_seed=None, agent_seed=None):
    runs, windows = compute_all_multiverses(env, actions, reward_df, debug, env_seed, agent_seed)
    if len(runs) == 0:
        return reward_df, windows
    multiverse_df = make_df_from_res(runs, debug, multiverse=True)
    return pd.concat([reward_df, multiverse_df], ignore_index=True, sort=False), windows


def compute_all_multiverses(env, actions, reward_df, debug=False, env_seed=None, agent_seed=None):
    runs = []
    windows = get_windows_from_df(reward_df)
    for window in windows:
        begin = int(window.split("_")[0])
        end = int(window.split("_")[1])
        for attack in windows[window]:
            attack_topos = windows[window][attack]['topos']
            attack = windows[window][attack]['attack']
            # get the list of topologies that where not computed [all_topo] - windows[u,v][A]
            universes = actions.copy()
            universes = filter(lambda x: get_action_name(x, debug) not in attack_topos, actions)
            # Remove indexes from already computed topologies, need to sort and reverse the indices to not messup the positions
            # for index in sorted(attack_topos, reverse=True):
            #    del universes[index]
            for universe in universes:
                run = compute_one_multiverse(env, universe, attack, begin, end, env_seed, agent_seed)
                runs.append(run)
    print("Number of multiverse computed :" + str(len(runs)))
    return runs, windows


def compute_one_multiverse(env, universe, attack, begin, end, env_seed=None, agent_seed=None):
    '''

    :param env:
    :param universe:
    :param attack:
    :param begin: WARNING : This is the begin timestep of the window, not the begin of the computation, and
                            from the point of vue of the Runner, not the chronix
                    If a window of attack is [44, 100] :
                    - this "begin" will be 44
                    - the chronic (real life) timestep will be 45
    :param end:
    :param env_seed:
    :param agent_seed:
    :return:
    '''
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        # TODO what do i do if agent cannot do opponent action ?
        # Retrieve line that is attacked
        line_id = attack.as_dict()['set_line_status']["disconnected_id"][0]
        action = universe.grid2op_action
        action.line_or_set_bus = [(line_id, 0)]
        action.line_ex_set_bus = [(line_id, 0)]
        if action.line_or_change_bus[line_id]:
            action.line_or_change_bus = [line_id]
        if action.line_ex_change_bus[line_id]:
            action.line_ex_change_bus = [line_id]
        combinated_action = action + attack
    agent = OneChangeThenOnlyReconnect.gen_next(combinated_action)(env.action_space)
    # set the seed
    env.chronics_handler.tell_id(-1)
    if env_seed is not None:
        env.seed(env_seed)
    obs = env.reset()
    if agent_seed is not None:
        agent.seed(agent_seed)
    # We fast forward a number of timestep from the point of vue of the chronic
    # (so begin + 1 to arrive at the correct time step to compute, since begin was already computed)
    env.fast_forward_chronics(begin+1)
    obs = env.current_obs
    agent.reset(obs)
    episode = init_episode_data(env, end - begin)
    reward = float(env.reward_range[0])
    done = False
    iteration = 0
    time_step = begin+1
    cum_reward = dt_float(0.0)
    while time_step <= end and not done:
        act = agent.act(obs, reward, done)
        obs, reward, done, info = env.step(act)
        cum_reward += reward
        opp_attack = attack
        iteration += 1
        episode.incr_store(True, iteration, 0,
                           float(reward), env._env_modification,
                           act, obs, None,
                           info)
        time_step += 1
    max_timestep = time_step-1
    nb_timestep = iteration
    episode.set_meta(env, iteration, float(cum_reward), env_seed, agent_seed)
    res = [(None, None, cum_reward, iteration, max_timestep, episode)]
    run = Run(universe, res, begin_ts=begin + 1)
    run.action = universe
    run.attacks = [attack for i in range(begin,end+1)]
    run.max_ts = end + 1
    run.reset_attacks_id()
    return run


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
