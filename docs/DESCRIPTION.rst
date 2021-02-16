***********
Description
***********

Introduction
============

An Oracle system is defined by it's ability to know all possible outcome,
in order to deduce the best course of actions. Named after an Oracle, who knows the future.

This is an Oracle System that tries to brute force the best possible combination of actions taken by an agent, in a certain Grid2OP environment and on a given episode.
It does so by using data from a set of user-fed actions played in a dummy environment that allows any actions.
It then finds the best possible course of actions that an agent can take, called "Best path".
Finally, a few KPIs are produced in order to give a quick rundown of the state of the environment initially provided.

This can allow you to test the boundaries of a given network environment, and get a better understanding of the potential weaknesses
in the decision making process of an agent

The following sections are dedicated to describe each step of the process of Oracle, the API features and the results it provides.
It will go through a didactic example to follow each step in a detailed manner

Process Overview
=======================

* **[Preliminary step] Easily create your own unitary actions visually**
    A notebook is provided in *oracle4grid/core/action_utils/Atomic_Actions_Helper.ipynb*. Follow the steps to visualize your grid and the impact of your unitary (atomic) actions on it thanks to grid2op plot API and dedicated functions in *core/utils/action_generator* that you'll learn to use.

    .. image:: images/notebook_screen.JPG

    You'll eventually write your unitary actions in a json format which will be directly usable by Oracle.

* **0 - Prepare environment and parse unitary actions**
    The grid2op environment for action simulations is prepared with provided parameters, the unitary actions are formatted to oracle format (3 formats are supported, a parser API is provided)
* **1 - Compute all unitary action combinations**
    We create OracleAction objects that correspond to the possible combinations of unitary actions (with a given maximum combination depth). It embeds a valid grid2op action format for simulation in next step.
    First filters are applied to actions if they cause divergence at the first timestep of the episode or if they don't have any effect on the initial topology
* **2 - Reward simulation**
    Each OracleAction is simulated by being applied alone on the grid, with possible parallel computation. We retrieve the rewards obtained at each timestep and the overloads - or any reward set in other_rewards
* **3 - Graph computation**
    From the previous simulation, a networkX graph is computed, its nodes being actions at given timesteps, each edge representing a possible transition from one action to the other, weighted by the corresponding reward
* **4 - Best path computation**
    The path that minimizes or maximizes the cumulated reward is calculated thanks to networkX API (Bellman-Ford algorithm and DAG longest path). 2 trajectories are provided: with or without possibility of overflows in the grid
* **5 - Indicators computation**
    Some useful indicators are computed. See details in **Indicators** section
* **6 - Trajectory replay in real game rules conditions**
    The best path is played by an OracleAgent, with a possible new set of game rules. It warns the user if these game rules lead to game over or diverging timesteps (if the expected cumulated reward is not matched)

All along this process, some objects are serialized if you chose debug=1. This will be detailed in section **Indicators**

Process detailed implementation
================================

.. image:: images/detailed_workflow.JPG


Didactic example
=================

In this section, we'll dig into the algorithm of Oracle with a simple example.

You'll be more familiar with the important objects of Oracle and see in particular those which are serialized in debug mode

Grid, actions and configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The grid is wcci_test which has 35 nodes (provided in *data/wcci_test/*)

Two unitary actions will be considered. They have been taken from the competition and are provided
in *oracle4grid/ressources/actions/wcci_test/2actions_winner_format2.json*. They follow the following format:

.. code-block:: python

 {"sub": {"1": [{"set_configuration": [0, 0, 0,..., 2, 0, 0, ..., 0, 0]}}}
 {"sub": {"23": [{"set_configuration": [0, 0, 0,..., 2, 0, 0, ..., 0, 0]}}}

Which is parsed by OracleParser to:*

 .. code-block:: python

     {'sub': {1: {'gens_id_bus': [[0, 1]],
                  'lines_id_bus': [[3, 2], [4, 2], [12, 1]],
                  'loads_id_bus': [[2, 1], [3, 2]]}}}
     {'sub': {23: {'gens_id_bus': [[11, 1], [12, 1], [13, 1]],
                   'lines_id_bus': [[30, 2],
                                    [31, 2],
                                    [32, 2],
                                    [34, 1],
                                    [37, 1],
                                    [38, 1]],
                   'loads_id_bus': [[24, 1]]}}}]

We can see that those two action have an impact on substations 1 and 23 respectively

In config.ini, we set maxIter to 4, to simulate 4 timesteps. We set maxDepth to 2 as we will only need to combine a maximum of 2 actions.

Action combinations
^^^^^^^^^^^^^^^^^^^^

We generate 4 OracleAction which have the following representation of atomic actions: 'sub-<id of substation>-<id of atomic action>'. When combinated, the atomic actions are separated by an underscore
``[sub-1-0, sub-23-1, sub-1-0_sub-23-1, donothing-0]``
In this step, actions can be filtered out if they cause a divergence at the first simulation timestep or if they don't have impact on the initial topology
There is no filtering needed here

.. image:: images/didactic_step1.JPG

Reward simulation
^^^^^^^^^^^^^^^^^^^^

Each OracleAction is applied on grid and the whole episode is then simulated in parallel by agent OneChangeThenOnlyReconnect

.. image:: images/didactic_step2.JPG

The resulting reward_df is a pandas.DataFrame representing the reward obtained at each timestep of those parallel simulation. it also includes whether there has been an overflow in the timestep (overload_reward = -1)

Graph computation
^^^^^^^^^^^^^^^^^^^^

A graph is computed thanks to the result of this simulation

.. image:: images/didactic_step3.JPG

The transition between actions (represented by the edges of the graph) are permitted or not according to the provided game rules.
These game rules are in constants.DICT_GAME_PARAMETERS_GRAPH

.. code-block:: python

    DICT_GAME_PARAMETERS_GRAPH = {'MAX_LINE_STATUS_CHANGED': 1,
                                  'MAX_SUB_CHANGED': 1}

Here you can see that one *substation maximum* can be impacted in each timestep, which is why ``sub-1-0_sub-23-1`` can't be applied in one timestep

Indicators
==============

Enumerate - hierarchy




