import unittest
from oracle4grid.main import load_and_run


CONFIG = {
    "max_depth": 3,
    "max_iter": 6,
    "nb_process": 1,
    "best_path_type": "shortest",
    "n_best_topos" : 2
}


class IntegrationTest(unittest.TestCase):
    def test_base_run(self):
        file = "./oracle4grid/ressources/actions/test_unitary_actions.json"
        chronic = "000"
        env_dir = "./oracle4grid/ressources/grids/rte_case14_realistic"
        res = load_and_run(env_dir, chronic, file, False, CONFIG)
        self.assertNotEqual(res, None)
        return 1

    def test_best_path_equal(self):
        file = "./oracle4grid/ressources/actions/test_unitary_actions.json"
        chronic = "000"
        env_dir = "./oracle4grid/ressources/grids/rte_case14_realistic"
        res = load_and_run(env_dir, chronic, file, False, CONFIG)
        best_path_actual = list(map(lambda x : x.__str__(), res))
        best_path_expected = ['sub-5-2', 'sub-1-1_sub-5-2', 'sub-1-1_sub-5-2', 'sub-1-1_sub-5-2', 'sub-1-1_sub-5-2', 'sub-1-1_sub-5-2']
        self.assertListEqual(best_path_actual, best_path_expected)



if __name__ == '__main__':
    unittest.main()
