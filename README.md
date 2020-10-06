# Oracle4Grid

This is a repository to compute an Oracle score on a scenario for a given Grid2op environment.
It finds the best course of actions aposteriori, within a given perimeter of actions (cf « Learning to run a power network challenge for training topology controllers »)

![Influence Graph](TransitionGraph_bestActions.png)

## Current content
3 steps and notebooks
- Create_Unitary_Actions_Viz.ipynb to create a dictionnary of unitary actions that are of interest and that will be combined after
- Prepare_for_Oracle.ipynb run the topology configurations in parallel, and save the rewards and topology configuration dataframes
- Analyse_Policy_Oracle-Scenario_3.ipynb Create the Transition Graph and find the best course of actions

## Workflow
![Influence Graph](OracleProcess.png)



