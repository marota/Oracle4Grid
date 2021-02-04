***************
Getting Started
***************

Launch Oracle4Grid
====================

To execute in **manual mode**, from root folder, type:

``pipenv run python -m main -d 0 -f "oracle4grid/ressources/actions/rte_case14_realistic/test_unitary_actions.json" -e "data/rte_case14_realistic" -c 0``

--debug | -d int
                            if 1, prints additional information for debugging purpose, but also serializes some result files in output folder (see dedicated chapter **Results**)
                            If 0, doesn't print detailed info
--file | -f string
                            File path to a json file containing atomic actions to be played. See dedicated chapter **Atomic actions**
--env | -e string
                            Path to directory containing the grid2op environment and its chronics
--chronic | -c
                            Name (string) or id (int) to the episode we want to consider. By default, the first available chronic will be chosen. Oracle4Grid only runs for one episode.
--agent_seed | -as
                            Agent seed to be used by the grid 2op runner. By default, None is considered.
--env_seed | -es
                            Environment seed to be used by the grid 2op runner. By default, None is considered.

See **Algorithm Description section** to learn more about the workflow and results.

Results
================

Returned by oracle
printed and returned in debug mode


Atomic actions
================

Describe rules and handled formats

Agent replay
================

OracleAgent to replay best path
Replat mode in oracle to check game overs

Configuration
===============

config.ini
In manual mode, further configuration is made through alphadeesp/config.ini

* *simulatorType* - you can chose Grid2op or Pypownet
* *gridPath* - path to folder containing files representing the grid
* *CustomLayout* - list of couples reprenting coordinates of grid nodes. If not provided, grid2op will load grid_layout.json in grid folder
* *grid2opDifficulty* - "0", "1", "2" or "competition". Be careful: grid datasets should have a difficulty_levels.json
* *7 other constants for alphadeesp computation* can be set in config.ini, with comments within the file

constants.py


Tests
=====

To launch the test suite:
``pipenv run python -m pytest --verbose --continue-on-collection-errors -p no:warnings``

