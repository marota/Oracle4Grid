#!/usr/bin/python3
__author__ = "NMegel, mjothy"

import os
import argparse
import configparser
import json

from oracle4grid.core.utils.prepare_environment import prepareParams, prepareEnv
from oracle4grid.core.oracle import oracle


def main():
    # ###############################################################################################################

    parser = argparse.ArgumentParser(description="Oracle4Grid")
    parser.add_argument("-d", "--debug", type=int,
                        help="If 1, prints additional information for debugging purposes. If 0, doesn't print any info", default=1)
    parser.add_argument("-f", "--file", type=str,
                        help="File path for the dict of atomic actions", default = "oracle4grid/ressources/actions/test_unitary_actions.json")
    parser.add_argument("-e", "--env", type=str,
                        help="Directory path for the environment to use", default="oracle4grid/ressources/grids/rte_case14_realistic")
    parser.add_argument("-c", "--chronic", type=str,
                        help="Name or id of chronic scenario to consider, as stored in chronics folder. By default, the first available chronic scenario will be chosen" , default=0)

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
    param = prepareParams() # Move to ini?
    env = prepareEnv(args.env, args.chronic, param)

    # Load unitary actions
    with open(args.file) as f:
        atomic_actions = json.load(f)

    # Run all steps
    oracle(atomic_actions, env, args.debug, config['DEFAULT'])
    return 1



if __name__ == "__main__":
    main()
