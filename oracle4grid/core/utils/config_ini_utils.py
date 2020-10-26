MAX_DEPTH = "max_depth"
MAX_ITER = "max_iter"
NB_PROCESS = "nb_process"

# Default config dict you may use for calling oracle.py directly
DEFAULT_CONFIG = {
    # Max atomic actions to combinate
    MAX_DEPTH: 3,
    # Max timestep to reach in episode
    MAX_ITER: 20,
    # Number of threads the computation engine is allowed to use
    NB_PROCESS: 1
}
