#!/usr/bin/python3
__author__ = "NMegel, mjothy"

import os
import argparse
import configparser

from oracle4grid.core.utils.launch_utils import load_and_run


def main():
    # ###############################################################################################################

    parser = argparse.ArgumentParser(description="Oracle4Grid")
    parser.add_argument("-d", "--debug", type=int,
                        help="If 1, prints additional information for debugging purposes. If 0, doesn't print any info", default=1)
    parser.add_argument("-f", "--file", type=str,
                        help="File path for the dict of atomic actions", default="oracle4grid/ressources/actions/rte_case14_realistic/test_unitary_actions_3.json")
    parser.add_argument("-e", "--env", type=str,
                        help="Directory path for the environment to use", default="data/rte_case14_realistic")
    parser.add_argument("-c", "--chronic", type=str,
                        help="Name or id of chronic scenario to consider, as stored in chronics folder. By default, the first available chronic scenario will be chosen",
                        default=0)
    parser.add_argument("-as", "--agent_seed", type=int,
                        help="Agent seed to be used by the grid2op runner",
                        default=None)
    parser.add_argument("-es", "--env_seed", type=int,
                        help="Environment seed to be used by the grid2op runner",
                        default=None)

    args = parser.parse_args()
    config = configparser.ConfigParser(allow_no_value=True)
    config.read("oracle4grid/ressources/config.ini")
    print("#### PARAMETERS #####")
    for key in config["DEFAULT"]:
        print("key: {} = {}".format(key, config['DEFAULT'][key]))
    print("#### ########## #####\n")

    if (args.file is None) or not os.path.exists(args.file):
        raise ValueError("Could not find file provided :" + str(args.file))

    if(args.agent_seed is not None):#Grid2op runner expect a list of seeds
        args.agent_seed=[args.agent_seed]
    if(args.env_seed is not None):
        args.env_seed=[args.env_seed]

    if args.debug > 1:
        raise ValueError("Input arg error, --debug, options are 0 or 1")
    else:
        args.debug = bool(args.debug)
    load_and_run(args.env, args.chronic, args.file, args.debug,args.agent_seed,args.env_seed, config['DEFAULT'])
    return 1


if __name__ == "__main__":
    main()
