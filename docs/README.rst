Mentions
=========

Quick Overview
------------------

An Oracle system is defined by it's ability to know all possible outcome,
in order to deduce the best course of actions. Named after an Oracle, who knows the future.

This is an Oracle System that tries to brute force the best possible combination of actions taken by an agent, in a certain Grid2OP environment.
It does so by using data from a set of user-fed actions played in a dummy environment that allows any actions.
It then finds the best possible course of actions that an agent can take, called "Best path".
Finally, a few KPIs are produced in order to give a quick rundown of the state of the environment initially provided.

This can allow you to test the boundaries of a given network environment, and get a better understanding of the potential weaknesses
in the decision making process of an agent

Features
----------

- Reads a json with a list of atomic actions (=unitary "action bricks" that affects a single substation or line) - It handles 3 different format to provide these atomic actions through an OracleParser object
- Creates a set of possible and grid2op-valid actions by computing any possible combination of the atomic actions (the resulting objects are called OracleActions) - *In the combinations, we allow only one atomic action per substation*
- Runs the simulation for all OracleActions, and stores the reward in a dataframe
- Generates a NetworkX graph with each node being a timestep in the aforementioned simulations - *The connection between two nodes is given by the gamerule*
- Runs a best path algorithm to deduce a series of action to achieve best results
- Runs an agent to try the chosen path
- Computes a few KPIs to better understand the path chosen, as well as cumulated rewards for comparison purposes

Contribute
-------------

- Issue Tracker: https://github.com/marota/Oracle4Grid/issues
- Source Code: https://github.com/marota/Oracle4Grid

Support
----------

If you are having issues, please let us know on github

License
---------
Copyright 2020-2021 RTE France

    RTE: http://www.rte-france.com

This Source Code is subject to the terms of the Mozilla Public License (MPL) v2.
