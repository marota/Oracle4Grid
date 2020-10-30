#!/usr/bin/python3
__author__ = "NMegel, mjothy"

import os
import argparse
import configparser
import json

from oracle4grid.core.utils.prepare_environment import prepare_params, prepare_env
from oracle4grid.core.oracle import oracle


def main():
    # ###############################################################################################################

    parser = argparse.ArgumentParser(description="Oracle4Grid")
    parser.add_argument("-d", "--debug", type=int,
                        help="If 1, prints additional information for debugging purposes. If 0, doesn't print any info", default=1)
    parser.add_argument("-f", "--file", type=str,
                        help="File path for the dict of atomic actions", default="oracle4grid/ressources/actions/test_unitary_actions_5.json")
    parser.add_argument("-e", "--env", type=str,
                        help="Directory path for the environment to use", default="oracle4grid/ressources/grids/rte_case14_realistic")
    parser.add_argument("-c", "--chronic", type=str,
                        help="Name or id of chronic scenario to consider, as stored in chronics folder. By default, the first available chronic scenario will be chosen",
                        default=0)

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read("oracle4grid/ressources/config.ini")
    print("#### PARAMETERS #####")
    for key in config["DEFAULT"]:
        print("key: {} = {}".format(key, config['DEFAULT'][key]))
    print("#### ########## #####\n")

    if (args.file is None) or not os.path.exists(args.file):
        raise ValueError("Could not find file provided :" + str(args.file))

    if args.debug > 1:
        raise ValueError("Input arg error, --debug, options are 0 or 1")
    else:
        args.debug = bool(args.debug)

    # ###############################################################################################################
    # Load Grid2op Environment with Parameters
    param = prepare_params()  # Move to ini?
    env = prepare_env(args.env, args.chronic, param)

    # Load unitary actions
    with open(args.file) as f:
        atomic_actions = json.load(f)

    # Init debug mode if necessary
    if args.debug:
        debug_directory = init_debug_directory(args)
    else:
        debug_directory = None

    # Run all steps
    oracle(atomic_actions, env, args.debug, config['DEFAULT'], debug_directory=debug_directory)
    return 1

def init_debug_directory(args):
    action_file = os.path.split(args.file)[len(os.path.split(args.file))-1].replace(".json","")
    grid_file = os.path.split(args.env)[len(os.path.split(args.env)) - 1]
    scenario = "scenario_"+str(args.chronic)
    debug_directory = os.path.join("oracle4grid/ressources/debug/", grid_file, scenario, action_file)
    os.makedirs(debug_directory, exist_ok=True)
    return debug_directory

if __name__ == "__main__":
    main()
