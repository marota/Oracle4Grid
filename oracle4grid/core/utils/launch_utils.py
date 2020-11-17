import json
import os

from oracle4grid.core.utils.prepare_environment import prepare_simulation_params, prepare_env
from oracle4grid.core.oracle import oracle


def load_and_run(env_dir, chronic, action_file, debug, config):
    # Load Grid2op Environment with Parameters
    param = prepare_simulation_params()  # Move to ini?
    env = prepare_env(env_dir, chronic, param)

    # Load unitary actions
    with open(action_file) as f:
        atomic_actions = json.load(f)

    # Init debug mode if necessary
    if debug:
        debug_directory = init_debug_directory(env_dir, action_file, chronic)
    else:
        debug_directory = None

    # Run all steps
    return oracle(atomic_actions, env, debug, config, debug_directory=debug_directory)


def init_debug_directory(env_dir, action_file, chronic):
    action_file_os = os.path.split(action_file)[len(os.path.split(action_file)) - 1].replace(".json", "")
    grid_file_os = os.path.split(env_dir)[len(os.path.split(env_dir)) - 1]
    scenario = "scenario_" + str(chronic)
    debug_directory = os.path.join("oracle4grid/output/", grid_file_os, scenario, action_file_os)
    os.makedirs(debug_directory, exist_ok=True)
    return debug_directory
