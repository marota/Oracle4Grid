MAX_DEPTH = "max_depth"
MAX_ITER = "max_iter"
NB_PROCESS = "nb_process"
N_TOPOS = "n_best_topos"

# Default config dict you may use for calling oracle.py directly
DEFAULT_CONFIG = {
    # Max atomic actions to combinate
    MAX_DEPTH: 3,
    # Max timestep to reach in episode
    MAX_ITER: 20,
    # Number of threads the computation engine is allowed to use
    NB_PROCESS: 1,
    # Number of topos in best path to compute in indicators
    N_TOPOS: 2
}
