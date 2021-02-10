# Oracle4Grid

This is a repository to compute an Oracle score on a scenario for a given Grid2op environment.
It finds the best course of actions aposteriori, within a given perimeter of actions (cf « Learning to run a power network challenge for training topology controllers »)

![Influence Graph](TransitionGraph_bestActions.png)

## Run

Run example

``
pipenv run python -m main -d 1 -f oracle4grid/ressources/actions/rte_case14_realistic/test_unitary_actions.json -e data/rte_case14_realistic -c 0
``

- -d/--debug: If 1, prints additional information for debugging purposes. If 0, doesn't print any info
- -f/--file: Directory path to .json file containing all atomic actions we want to play 
- -e/--env: Directory path or name of the Grid2op environment to load
- -c/--chronic: Name or id of chronic scenario to consider, as stored in chronics folder. By default, the first available chronic scenario will be chosen

For more customisation options, see section "Configuration for the user" below

## Installation

For development, please proceed as follows:
- Create a pipenv environment with required dependencies. It will install from the Pipfile
``pipenv install
``
- [Optional] Install lightsim2grid to speed up powerflow simulation. Instructions at: https://github.com/BDonnot/lightsim2grid. Otherwise, you can use PandaPowerBackend.
- To generate a jupyter kernel from this environment
``
pipenv run ipython kernel install --user --name=<YourEnvName>
``
- [Optional] Enable tqdm to paralelize in jupyter notebooks
``
jupyter nbextension enable --py widgetsnbextension
``

- [TEMPORARY]  Install the forked grid2op
Checkout the forked grid2op version, NOT in the oracle4grid repository

`git clone https://github.com/mjothy/Grid2Op.git`

`git checkout -b mj-devs`

`cd Grid2Op/`

`pip install -U .`

## Tests

There are a lot of integration tests already implemented, one can run them with :

`python -m pytest oracle4grid/test --verbose --continue-on-collection-errors -p no:warnin`

## Configuration for the user

### Config.ini
The main configuration of the oracle module is located in ./oracle4grid/ressources/config.ini
This file contains the following options :

- Max atomic actions to combinate
max_depth = 5

- Max timestep to reach in episode
max_iter = 5

- Number of significant digits for reward, don't write value if you don't want to truncate and the parameter will be None
reward_significant_digit

- Number of threads the computation engine is allowed to use
nb_process = 1

- Computation type for finding path "shortest|longest"
best_path_type = longest

- Number of topos in best path to compute in indicators
n_best_topos = 2

### Atomic actions

Oracle requires a base of atomic actions which are provided in .json format.
These unitary elements plays the role of action "bricks" that will be combined by Oracle to build more consistent actions on the grid.

As a limitation, two types of atomic actions are handled, each one having a standard format:

* An atomic action which impact **one** substation topology - i.e. setting buses of assets (lines origins, lines extremities, generators, loads)
    ``{"sub": {"1": [{"lines_id_bus": [[0, 2], [2, 2]], "loads_id_bus": [[0, 2]], "gens_id_bus": [[0,2]]}}``
* An atomic actions that disconnects **one** line
    ``{"line": {"4": [{"set_line": -1}]}}``

A user-friendly notebook is provided to help the user defining atomic actions and visualize their impact on the grid. See *oracle4grid/core/actions_utils/Atomic_Actions_Helper.ipynb*

2 other formats are handled by class *oracle4grid/core/utils/launch_utils::OracleParser*. It should be easy for a user to develop an additional parser to handle his action format using the same API

* An explicit format for substation topologies (parser1) - examples in *oracle4grid/ressources/neurips_track1*
    ``{"set_bus": {"substations_id": [[16, [1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2]]]}}``
* An explicit format for whole action space (parser2) - examples in *oracle4grid/ressources/wcci_test*
    ``{"sub": {"1": [{"set_configuration": [0, 0, 0,..., 2, 0, 0, ..., 0, 0]}}}``



### constants.py
In addition to the ini file, there is a constant API available for easy local overrides of some common behaviors / implementations.
This API can be used in two ways :
- Local override :
You may change the file itself to experiment with the different parameters of the API. See the comments in the file.
- API override :
The main function (load_and_run() and oracle()) of the Oracle allow for a constants argument that you can pass. It is usually an instance of a sub-class of the default implementation

## Current content
3 steps and notebooks
- Create_Unitary_Actions_Viz.ipynb to create a dictionnary of unitary actions that are of interest and that will be combined after
- Prepare_for_Oracle.ipynb run the topology configurations in parallel, and save the rewards and topology configuration dataframes
- Analyse_Policy_Oracle-Scenario_3.ipynb Create the Transition Graph and find the best course of actions

## Workflow
![Influence Graph](OracleProcess.png)



