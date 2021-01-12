import os
import warnings

from grid2op.Agent import OneChangeThenNothing
from grid2op.Runner import Runner
from grid2op.Environment import Environment
from oracle4grid.core.reward_computation.Run import Run

from oracle4grid.core.agent.OracleAgent import OracleAgent

from oracle4grid.core.utils.constants import ENV_SEEDS, AGENT_SEEDS
from oracle4grid.core.utils.prepare_environment import prepare_replay_params, prepare_env, prepare_game_params, prepare_simulation_params

def set_replay_parameters(env):
    params = prepare_replay_params()
    env.parameters = params
    return env


def replay(action_path: list, max_iter: int,
           kpis, grid_path, chronic_id, debug = False):
    if debug:
        print('\n')
        print("============== 6 - Replay OracleAgent on best path with real condition game rules ==============")
    # Environment settings for replay
    param = prepare_replay_params()
    env = prepare_env(grid_path, chronic_id, param)
    env.set_id(chronic_id)
    obs = env.reset()


    # Run replay
    agent = OracleAgent(action_path=action_path, action_space=env.action_space,
                        observation_space=None, name=None)
    agent_reward = 0.
    done = False
    for t in range(max_iter):
        if done:
            warnings.warn("During replay - oracle agent has game over before max iter (timestep "+str(t)+") with exception: "+str(info['exception']))
            break
        action = agent.act(obs, reward=0., done=False)
        obs, reward, done, info = env.step(action) # obs.simulate(action, time_step = 0)
        agent_reward += reward

    # Check reward as expected
    expected_reward = extract_expected_reward(kpis)
    if expected_reward != agent_reward and t==(max_iter-1):
        warnings.warn("During replay - oracle agent does not retrieve the expected reward. Some timestep may have break some game rules in real condition. Expected reward: "+str(expected_reward)+" Reward obtained: "+str(agent_reward))
    elif t==(max_iter-1):
        print("Expected reward of "+str(expected_reward)+" has been correctly obtained in replay conditions")
    return t

def extract_expected_reward(kpis):
    return kpis.loc[kpis['Indicator name']=="Best possible path with game rules", "Reward value"].values[0]