#!/usr/bin/python3
__author__ = "NMegel, mjothy"

import os
import argparse
import configparser


def main():
    # ###############################################################################################################

    parser = argparse.ArgumentParser(description="Oracle4Grid")
    parser.add_argument("-d", "--debug", type=int,
                        help="If 1, prints additional information for debugging purposes. If 0, doesn't print any info", default=0)
    parser.add_argument("-f", "--file", type=str,
                        help="File path for the dict of atomic actions")
    parser.add_argument("-e", "--env", type=str,
                        help="Directory path for the environment to use")

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read("./resources/config.ini")
    print("#### PARAMETERS #####")
    for key in config["DEFAULT"]:
        print("key: {} = {}".format(key, config['DEFAULT'][key]))
    print("#### ########## #####\n")

    if (args.file is None) or os.path.exists(args.file):
        raise ValueError("Could not find file provided :" + args.file)

    if args.snapshot > 1:
        raise ValueError("Input arg error, --snapshot, options are 0 or 1")

    if args.debug > 1:
        raise ValueError("Input arg error, --debug, options are 0 or 1")

    # ###############################################################################################################
    #run all steps
    return 1


if __name__ == "__main__":
    main()
