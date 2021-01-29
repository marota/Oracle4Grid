Installation
------------

1. First clone the repos
^^^^^^^^^^^^^^^^^^^^^^^^

``git clone https://github.com/marota/Oracle4Grid.git``


2. Install python packages
^^^^^^^^^^^^^^^^^^^^^^^^^^

``pip3 install (-U) .``

or

``pipenv install (-U) .``

3. (Optional) Change the configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main configuration of the oracle module is located in ./oracle4grid/ressources/config.ini
You may change some settings here depending on your needs.

4. Run with your dataset
^^^^^^^^^^^^^^^^^^^^^^^^

``cd ./Oracle4grid``

``python -m main.py -f ./oracle4grid/ressources/actions/test_unitary_actions.json -e  ./oracle4grid/ressources/grids/rte_case14_realistic -c 000``

5. (Optional) Compile and output the sphinx doc (this documentation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run
``./docs/make.bat html``
