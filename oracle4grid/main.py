#!/usr/bin/python3
__author__ = "NMegel, mjothy"

import os
import argparse
import configparser

from oracle4grid.core.oracle import load_and_run


def main():
    # ###############################################################################################################

    parser = argparse.ArgumentParser(description="Oracle4Grid")
    parser.add_argument("-d", "--debug", type=int,
                        help="If 1, prints additional information for debugging purposes, but also serializes some result files in output folder. If 0, doesn't print any info", default=1)
    parser.add_argument("-f", "--file", type=str,
                        help="File path to a json file containing atomic actions to be played", default="oracle4grid/ressources/actions/rte_case14_realistic/test_unitary_actions.json")
    parser.add_argument("-e", "--env", type=str,
                        help="Path to directory containing the grid2op environment and its chronics", default="data/rte_case14_realistic")
    parser.add_argument("-c", "--chronic", type=str,
                        help="Name or id of chronic scenario to consider, as stored in chronics folder. By default, the first available chronic scenario will be chosen",
                        default=0)
    parser.add_argument("-as", "--agent_seed", type=int,
                        help="Agent seed to be used by the grid2op runner",
                        default=None)
    parser.add_argument("-es", "--env_seed", type=int,
                        help="Environment seed to be used by the grid2op runner",
                        default=None)
    parser.add_argument(
        "--config-path",
        default=None,
        required=False,
        type=str,
        help="Path to the configuration file config.ini.",
    )
    args = parser.parse_args()

    pkg_root_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(pkg_root_dir,"ressources/config.ini")
    config = configparser.ConfigParser(allow_no_value=True)
    if args.config_path is not None:
        config.read(args.config_path)
    else:
        config.read(default_config_path)

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
    load_and_run(args.env, args.chronic, args.file, args.debug,args.agent_seed,args.env_seed, config['DEFAULT'])
    return 1


if __name__ == "__main__":
    main()
