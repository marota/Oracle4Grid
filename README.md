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

## Current content
3 steps and notebooks
- Create_Unitary_Actions_Viz.ipynb to create a dictionnary of unitary actions that are of interest and that will be combined after
- Prepare_for_Oracle.ipynb run the topology configurations in parallel, and save the rewards and topology configuration dataframes
- Analyse_Policy_Oracle-Scenario_3.ipynb Create the Transition Graph and find the best course of actions

## Workflow
![Influence Graph](OracleProcess.png)



