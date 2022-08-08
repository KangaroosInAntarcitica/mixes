import unittest
from .utils import *


class TestDGMM(unittest.TestCase):
    def test_path_permutations(self):
        layers = [2, 3]
        expected = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
        assert (get_paths_permutations(layers) == expected).all()

    def test_path_permutations_deep(self):
        layers = [2, 2, 3]
        expected = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                             [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                             [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]]).T
        assert (get_paths_permutations(layers) == expected).all()


if __name__ == "__main__":
    unittest.main()
